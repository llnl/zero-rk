
import numpy
import scipy.optimize
from matplotlib import pyplot as plt

#!TODO: Make into module and add driver script
#!TODO: Make better plots
#!TODO: User set T_range
#!TODO: User set parameter bounds
#!TODO: Unit conversion


class negative_rate_refitter(object):
    def __init__(self, mech_file):
        self.prelines = []
        self.postlines = []
        self.reaction_header_lines = []
        self.reactions = []
        self.reaction_eqns = []
        self.duplicates = dict()
        self.opt_rxn_eqns = set()
        self.T_range = [300, 3500]
        self.CONVERT = (4184.0/8.3144598e3)
        self.mech_file=mech_file
        self.parse(mech_file)
        self.opt_rxns()

    def parse(self, mech_file):
        with open(mech_file,'r', encoding='utf8') as mf:
            self.lines = mf.readlines()
        reaction_section=False
        post_reaction_section=False
        for line in self.lines:
            comment = "!".join(line.split("!")[1:])
            linex = line.split("!")[0].strip()
            if 'REAC' == line.strip().upper()[0:4]:
                reaction_section=True
                self.prelines.append(line)
                continue
            if reaction_section:
                if 'END' == line.strip().upper()[0:3]:
                    reaction_section=False
                    post_reaction_section=True
                    self.postlines.append(line)
                    continue
                if "=" in linex:
                    tokens=linex.split()
                    A, n, Ea = map(float,tokens[-3:])
                    eqn = ''.join(tokens[:-3])
                    rxn = {'eqn': eqn, 'A': A, 'n': n, 'Ea': Ea, 'aux': []}
                    rxn.update({'Aorig':A, 'norig': n, 'Eaorig':Ea})
                    rxn['comment'] = comment
                    rxn['aux_comments'] = []
                    rxn['line'] = line
                    rxn['deactivated'] = False
                    #Only optimizing if we have negative A values and oll duplicates together
                    if(A < 0 and eqn not in self.opt_rxn_eqns):
                        self.opt_rxn_eqns.add(eqn)
                    if eqn in self.reaction_eqns:
                        idxs = [i for i, x in enumerate(self.reaction_eqns) if x == eqn]
                        self.duplicates[eqn] = idxs + [len(self.reactions)]
                    self.reaction_eqns.append(rxn['eqn'])
                    self.reactions.append(rxn)
                elif(len(self.reactions)==0):
                    #Comments at head of REACTION section
                    self.reaction_header_lines.append(line)
                else:
                    self.reactions[-1]['aux'].append(line)
                    self.reactions[-1]['aux_comments'].append(comment)
            elif post_reaction_section:
                self.postlines.append(line)
            else:
                self.prelines.append(line)

    def opt_rxns(self):
        kmult = negative_rate_refitter._kmult
        T = numpy.linspace(self.T_range[0], self.T_range[1], 200)
        for eqn in self.opt_rxn_eqns:
            print(f"{eqn}")
            assert(eqn in self.duplicates)
            eqn_idxs = self.duplicates[eqn]

            n_old = len(eqn_idxs)

            p = []
            for idx in eqn_idxs:
                rxn = self.reactions[idx]
                #Ta = Ea * CONVERT
                p.extend( [rxn['A'], rxn['n'], rxn['Ea']*self.CONVERT] )

            K_orig = kmult(T,p,scale=False)
            popt=None
            for n_fit in [1,2,3]:
                opt_bounds = [ (1,20), (-5.0, 5.0), (0.2, 8) ]*n_fit
                opt_result = scipy.optimize.differential_evolution(negative_rate_refitter._opt_fn,  opt_bounds,
                                                                   args=(T, K_orig),
                                                                   popsize=240,
                                                                   maxiter=1000,
                                                                   seed=123,
                                                                   tol=1.0e-4,
                                                                   polish=True)
                popt = opt_result.x
                rmse = opt_result.fun

                max_err = numpy.max(numpy.abs(kmult(T,popt) - K_orig)/K_orig)

                print(f"{n_fit}: {max_err}, {popt}")
                plt.figure()
                ax = plt.gca()
                ax.set_xlabel("1000/T [1/K]")
                ax.set_ylabel("k [1/(mol*s)]")

                for i in range(n_old):
                    K = kmult(T, p[i*3:(i+1)*3], scale=False)
                    if(p[0+i*3] > 0):
                        ax.semilogy(1000/T, K, label=f"k{i+1}")
                    else:
                        ax.semilogy(1000/T, -K, label=f"-k{i+1}")

                Korig = kmult(T,p,scale=False)
                ax.semilogy(1000/T, Korig, label="$k_{sum}$")
                Knew = kmult(T,popt)
                ax.semilogy(1000/T, Knew, linestyle="--", label="$k_{fit}$")
                #ax.text(0.15, 0.9, f"$k_{{fit}} = {as_si(Afit,3)} T^{{{nfit:0.4g}}} e^{{ -{{{Tafit:0.4g}}}/RT }}$",
                #        size=12, transform=ax.transAxes)
                plt.legend()

                if(max_err < 0.05):
                    break
            assert(popt is not None)
            n_new = len(popt) // 3
            n_iter = max(n_new, n_old)
            # overwrite new reactions (and sneak new ones in if necessary)
            for i in range(n_iter):
                if(i < n_new and i < n_old):
                    orig_idx = eqn_idxs[i]
                    old_rxn = self.reactions[orig_idx]
                    Aorig, norig, Eaorig, comment = old_rxn['A'], old_rxn['n'], old_rxn['Ea'], old_rxn['comment']
                    if(not comment): comment = "\n"

                    self.reactions[orig_idx]['comment'] = f" ZERORKNEGREFIT {Aorig} {norig} {Eaorig} ! {comment}"
                    self.reactions[orig_idx]['A'] = numpy.power(10,popt[0+i*3])
                    self.reactions[orig_idx]['n'] = popt[1+i*3]
                    self.reactions[orig_idx]['Ea'] = numpy.power(10,popt[2+i*3])/self.CONVERT
                    if(n_new == 1): #remove duplicate from aux (if it's there)
                        for iaux, aux in enumerate(self.reactions[orig_idx]['aux']):
                            if(aux.strip().lower()[0:3] == 'dup'):
                                del self.reactions[orig_idx]['aux'][iaux]
                                break
                elif(i < n_new):
                    last_old_idx = eqn_idxs[-1]
                    A  = numpy.power(10,popt[0+i*3])
                    n  = popt[1+i*3]
                    Ea = numpy.power(10,popt[2+i*3])/self.CONVERT

                    self.reactions[last_old_idx]['aux'].append(f"{eqn} {A:0.4e} {n:0.6g} {Ea:0.6g} ! ZERORKNEGREFIT (new reaction)\n")
                    self.reactions[last_old_idx]['aux'].append("DUPLICATE\n")
                else:
                    assert(i < n_old and i >= n_new)
                    self.reactions[eqn_idxs[i]]['deactivated'] = True
                    old_comment = self.reactions[eqn_idxs[i]]['comment']
                    self.reactions[eqn_idxs[i]]['comment'] = f" ZERORKNEGREFIT (disabled) ! {old_comment}"


    @staticmethod
    def _opt_fn(p, T, kfit):
        kprop = negative_rate_refitter._kmult(T, p)
        return 1/len(T)*numpy.sum(numpy.power(numpy.abs(kfit - kprop)/kfit,2))

    @staticmethod
    def _kmult(T, p, scale=True):
        res = numpy.zeros(T.shape)
        for i in range(len(p)//3):
            A, n, Ta = p[3*i:3*(i+1)]
            if(scale):
                A = numpy.power(10,A)
                Ta = numpy.power(10,Ta)
            res += A*T**n*numpy.exp(-Ta/T)
        return res

    def write_mech(self, mech_file):
        with open(mech_file,'w') as mfile:
           for line in self.prelines:
               mfile.write(line)
           for line in self.reaction_header_lines:
               mfile.write(line)
           for idx, rxn in enumerate(self.reactions):
               eqn = self.reactions[idx]['eqn']
               c = ""
               if(self.reactions[idx]['deactivated']): c="!"
               if eqn in self.opt_rxn_eqns:
                   mfile.write(f"{c}{rxn['eqn']} {rxn['A']:0.4e} {rxn['n']:0.6g} {rxn['Ea']:0.6g}")
                   if len(rxn['comment']) > 0:
                       mfile.write(" !" + rxn['comment'])
                   else:
                       mfile.write("\n")
               else:
                   mfile.write(rxn['line'])
               for l in rxn['aux']:
                   mfile.write(c+l)
           for line in self.postlines:
               mfile.write(line)

    

if __name__ == "__main__":
    rf = negative_rate_refitter('mech.dat')
    rf.write_mech('mech_refit.dat')
    plt.show()


