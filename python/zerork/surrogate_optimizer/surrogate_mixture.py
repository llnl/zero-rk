
import sys
import os
import dill
import time
import subprocess
import numpy
from scipy import optimize
from ruamel.yaml import YAML

from ..config import ZERORK_ROOT

class SurrogateSpecies:
    ATOMIC_WEIGHTS={'C':12,'H':1,'O':16,'N':14}
    JOBACK_GROUPS=[ 'P', 'S', 'T', 'Q','Ps', 'Ss', 'Ts', 'Tss', 'Pss', 
                    'Sss', 'Sr', 'Srs', 'Trs', 'OH', 'O', 'Or',  'C=O',
                    'C=Or', 'HC=O', 'COO']
    def __init__(self, keys, db_line):
        words = db_line.split(',')
        words = words[0:len(keys)] #Strip off un-named keys/comments
        for key, word in zip(keys,words):
            key  = key.strip()
            if len(key) == 0:
                continue
            word = word.strip()
            try:
                word = float(word)
            except:
                pass
            vars(self).update({key: word})

        if self.kind not in ['paraffin','iso-paraffin','olefin','naphthene','aromatic',
                           'ethanol','alcohol','ester','furan','ether','ketone']:
            print("Unrecognized species kind \"{}\".".format(self.kind))
            sys.exit(1)

        self.carbon_number = int(self.getAtomNum("C"))
        if "CT1" in vars(self):
            self.carbon_type=numpy.zeros( (14,) )
            for ct in range(14):
                try:
                    ct_val = vars(self)["CT{}".format(ct+1)]
                    self.carbon_type[ct] = ct_val
                except:
                    break
        else:
            self.carbon_type = None

        if "AT0" in vars(self):
            self.atom_type=[]
            for at in range(18):
                try:
                    at_val = vars(self)["AT{}".format(at)]
                    self.atom_type.append(at_val)
                except:
                    break
            self.atom_type = numpy.array(self.atom_type)
        else:
            self.atom_type = None

        num_joback_parsed = 0
        joback_groups = [0]*len(self.JOBACK_GROUPS)
        for idx, jg in enumerate(self.JOBACK_GROUPS):
            if jg in vars(self):
                num_joback_parsed += 1
                jg_val = vars(self)[jg]
                joback_groups[idx] = jg_val

        if num_joback_parsed == len(self.JOBACK_GROUPS):
            self.joback_groups = numpy.array(joback_groups)
        else:
            if num_joback_parsed > 0:
                print("WARNING: Database Joback groups didn't match parser list.  Discarding.")
            self.joback_groups = None

        self.MM = 0.0
        for atom in self.ATOMIC_WEIGHTS.keys():
            self.MM += self.getAtomNum(atom)*self.ATOMIC_WEIGHTS[atom]

    def getAtomNum(self, atomStr):
        atomStr += "-atoms"
        return vars(self).get(atomStr,0.0)
    def calcDBE(self):
        C = self.getAtomNum("C")
        H = self.getAtomNum("H")
        dbe = (2*C+2-H)/2
        return dbe

class SurrogateMixture:
    def __init__(self, database, mech_file, therm_file):
        self.db = database
        self.mf = mech_file
        self.tf = therm_file
        self.jobname = "surrogate_optimizer"
        self.logfile = None
        self.species = []
        self.active_species_names = []
        self.num_species = 0
        self.best_val = numpy.inf
        self.printed_best = numpy.inf
        self.print_ratio = 0.005
        self.num_evals = 0
        self.last_print = 0
        self.print_interval = 1000
        self.min_mole_frac = 0.0
        self.mehl_distillation_alpha = 0.08
        self.zerork_exe = self._get_zerork_exe()
        self.zerork_mpi_exe = self._get_zerork_mpi_exe()
        self.dist_curve_data = None
        self.setTransform("alr")

        self._readDatabase()
        self.spA = numpy.array([sp.A for sp in self.species])
        self.spB = numpy.array([sp.B for sp in self.species])
        self.spC = numpy.array([sp.C for sp in self.species])
        self.spMV = numpy.array([sp.MM/sp.density for sp in self.species])

        self._checkMech()
        self._checkTherm()

    def _readDatabase(self):
        with open(self.db, 'r') as infile:
            header = infile.readline() # read column headings
            keys = header.split(",")
            for line in infile:
                line = line.strip()
                if len(line) == 0: continue #skip blank line
                self.addSpecies(SurrogateSpecies(keys,line))

    def _checkMech(self):
        if self.mf is None:
            return
        if not os.path.isfile(self.mf):
            print("Mechanism file not found: "+self.mf)
            sys.exit()

        #Check all database species are in mechanism file
        species_list = []
        with open(self.mf, 'r') as infile:
            species_start = False
            species_end = False
            for line in infile:
                if line.strip().lower()[0:4] == "spec":
                    species_start = True
                if species_start:
                    if line.strip().lower()[0:3] == "end":
                        species_end = True
                    species_list.extend(line.split())
                if species_end:
                    break
        for sp in self.species:
            if sp.name not in species_list:
                print("WARNING: Database species not found in mechanism: " + sp.name)

    def _checkTherm(self):
        if self.tf is None:
            return
        if not os.path.isfile(self.tf):
            print("Thermodynamics file not found: "+self.tf)
            sys.exit()
        #with open(self.tf, 'r') as infile:
           #Check all database species are in thermo file

    def addSpecies(self,specie):
        self.species.append(specie)
        self.num_species += 1

    def getSpeciesIndices(self, species_name_list):
        indices = []
        for name in species_name_list:
            found = False
            for idx, sp in enumerate(self.species):
                if sp.name == name:
                    found = True
                    indices.append(idx)
            assert(found)
        return indices

    def getVolumeFractions(self):
        return [sp.volume_fraction for sp in self.species]

    def getVolumeFractionsDict(self):
        vf_dict = {}
        for sp in self.species:
            if(sp.volume_fraction > 0.0):
                vf_dict[sp.name] = sp.volume_fraction
        return vf_dict

    def getMoleFractions(self):
        return self.getMoleFractionsFromVolumeFractions(self.getVolumeFractions())

    def getMoleFractionsDict(self):
        species_names = [sp.name for sp in self.species]
        species_mole_fracs = self.getMoleFractions()
        mf_dict = {}
        for spn, spmf in zip(species_names,species_mole_fracs):
            if(spmf > 0.0):
                mf_dict[spn] = spmf
        return mf_dict

    def getMassFractions(self):
        return self.getMassFractionsFromVolumeFractions(self.getVolumeFractions())

    def getMassFractionsDict(self):
        species_names = [sp.name for sp in self.species]
        species_mass_fracs = self.getMassFractions()
        mf_dict = {}
        for spn, spmf in zip(species_names,species_mass_fracs):
            if(spmf > 0.0):
                mf_dict[spn] = spmf
        return mf_dict

    def getCarbonNumbers(self):
        return [sp.carbon_number for sp in self.species]

    def getCarbonTypes(self):
        return [sp.carbon_type for sp in self.species]

    def getAtomTypes(self):
        return [sp.atom_type for sp in self.species]

    def getJobackGroups(self):
        return [sp.joback_groups for sp in self.species]

    def setVolumeFractionsArray(self,volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        for vf, sp in zip(volume_fractions, self.species):
            sp.volume_fraction = vf

    def setMoleFractionsDict(self,mf_dict):
        self.active_species_names = []
        for sp in self.species:
            sp.volume_fraction = 0.0
        mf_array = numpy.zeros(len(self.species))
        for key in mf_dict:
            self.active_species_names.append(key)
            found = False
            for isp,sp in enumerate(self.species):
                if key == sp.name:
                    found = True
                    mf_array[isp] = mf_dict[key]
            if not found:
                print("Error: Input composition species not in database: " + key)
                sys.exit()
        vf_array = self.getVolumeFractionsFromMoleFractions(mf_array)
        self.setVolumeFractionsArray(vf_array)

    def setMassFractionsDict(self,mf_dict):
        self.active_species_names = []
        for sp in self.species:
            sp.volume_fraction = 0.0
        mf_array = numpy.zeros(len(self.species))
        for key in mf_dict:
            self.active_species_names.append(key)
            found = False
            for isp,sp in enumerate(self.species):
                if key == sp.name:
                    found = True
                    mf_array[isp] = mf_dict[key]
            if not found:
                print("Error: Input composition species not in database: " + key)
                sys.exit()
        vf_array = self.getVolumeFractionsFromMassFractions(mf_array)
        self.setVolumeFractionsArray(vf_array)

    def setVolumeFractionsDict(self,vf_dict):
        self.active_species_names = []
        for sp in self.species:
            sp.volume_fraction = 0.0
        for key in vf_dict:
            self.active_species_names.append(key)
            found = False
            for sp in self.species:
                if key == sp.name:
                    found = True
                    sp.volume_fraction = vf_dict[key]
            if not found:
                print("Error: Input composition species not in database: {}".format(key))
                sys.exit()

    def getVolumeFractionsFromMassFractions(self, mass_fractions):
        assert(len(mass_fractions) == self.num_species)
        sp_specific_volume = [mf/sp.density for mf, sp in zip(mass_fractions,self.species)]
        total_specific_volume = numpy.sum(sp_specific_volume)
        volume_fractions = [ssv/total_specific_volume for ssv in sp_specific_volume]
        return volume_fractions

    def getVolumeFractionsFromMoleFractions(self, mole_fractions):
        assert(len(mole_fractions) == self.num_species)
        mass_fractions = self.getMassFractionsFromMoleFractions(mole_fractions)
        return self.getVolumeFractionsFromMassFractions(mass_fractions)

    def getMassFractionsFromVolumeFractions(self, volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        sp_mass = [vf*sp.density for vf, sp in zip(volume_fractions,self.species)]
        total_mass = numpy.sum(sp_mass)
        mass_fractions = [sm/total_mass for sm in sp_mass]
        return mass_fractions

    def getMassFractionsFromMoleFractions(self, mole_fractions):
        assert(len(mole_fractions) == self.num_species)
        sp_mass = [mf*sp.MM for mf, sp in zip(mole_fractions,self.species)]
        total_mass = numpy.sum(sp_mass)
        mass_fractions = [sm/total_mass for sm in sp_mass]
        return mass_fractions

    def getMolarVolumeFromMoleFractions(self, mole_fractions):
        assert(len(mole_fractions) == self.num_species)
        #Molar Volume = sum(mole_fractions*MW/density)
        return numpy.inner(numpy.array(mole_fractions), self.spMV)

    def getMoleFractionsFromVolumeFractions(self, volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        sp_moles = [vf*sp.density/sp.MM for vf, sp in zip(volume_fractions,self.species)]
        total_moles = numpy.sum(sp_moles)
        mole_fractions = [sm/total_moles for sm in sp_moles]
        return mole_fractions

    def calcHCRatio(self):
        volume_fractions = self.getVolumeFractions()
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        H = 0.0
        C = 0.0
        for mf,sp in zip(mole_fractions,self.species):
            H += sp.getAtomNum('H')*mf
            C += sp.getAtomNum('C')*mf
        return H/C

    def calcAtomFractions(self):
        volume_fractions = self.getVolumeFractions()
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        H = 0.0
        C = 0.0
        O = 0.0
        for mf,sp in zip(mole_fractions,self.species):
            H += sp.getAtomNum('H')*mf
            C += sp.getAtomNum('C')*mf
            O += sp.getAtomNum('O')*mf
        return (H, C, O)

    def calcDensity(self):
        volume_fractions = self.getVolumeFractions()
        density = 0.0
        for vf, sp in zip(volume_fractions, self.species):
            density += vf*sp.density
        return density

    def calcHCRatioFromVolumeFractions(self, volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        H = 0.0
        C = 0.0
        for mf,sp in zip(mole_fractions,self.species):
            H += sp.getAtomNum('H')*mf
            C += sp.getAtomNum('C')*mf
        return H/C

    def calcMixtureHoV(self):
        volume_fractions = self.getVolumeFractions()
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        mix_hov = numpy.sum([mf*sp.HoV for mf, sp in zip(mole_fractions, self.species)])
        return mix_hov

    def calcMixtureHoVMass(self):
        volume_fractions = self.getVolumeFractions()
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        mix_hov = numpy.sum([mf*sp.HoV for mf, sp in zip(mole_fractions, self.species)])
        molar_mass = numpy.sum([mf*sp.MM for mf, sp in zip(mole_fractions, self.species)])
        mix_hov_mass = mix_hov / molar_mass
        return mix_hov_mass

    def calcMolarMass(self):
        volume_fractions = self.getVolumeFractions()
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        molar_mass = numpy.sum([mf*sp.MM for mf, sp in zip(mole_fractions, self.species)])
        return molar_mass

    def getKindVolumeFraction(self,kind):
        volume_fractions = self.getVolumeFractions()
        kinds = kind.split("|")
        kvf = 0.0
        for vf, sp in zip(volume_fractions,self.species):
            if sp.kind in kinds or sp.name in kinds:
                kvf += vf
        return kvf

    def getKindMassFraction(self,kind):
        volume_fractions = self.getVolumeFractions()
        mass_fractions = self.getMassFractionsFromVolumeFractions(volume_fractions)
        kinds = kind.split("|")
        kmf = 0.0
        for vf, sp in zip(mass_fractions,self.species):
            if sp.kind in kinds or sp.name in kinds:
                kmf += vf
        return kmf

    def print_fun(self,msg):
        if(self.stdout):
            print(msg)
        if(self.logout):
            self.logfile.write(msg+"\n")

    def printVolumeFractions(self,volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        self.print_fun("{:>18}{:>18}".format("Species","Volume Fraction"))
        self.print_fun("{}".format("-"*36))
        for vf,sp in zip(volume_fractions,self.species):
            if(sp.name in self.active_species_names):
                self.print_fun("{:>18}{:>18.5g}".format(sp.name,vf))
        self.print_fun("")

    def printMoleFractions(self,volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        self.print_fun("{:>18}{:>18}".format("Species","Mole Fraction"))
        self.print_fun("{}".format("-"*36))
        for mf,sp in zip(mole_fractions,self.species):
            if(sp.name in self.active_species_names):
                self.print_fun("{:>18}{:>18.5g}".format(sp.name,mf))
        self.print_fun("")

    def printFractions(self,volume_fractions):
        assert(len(volume_fractions) == self.num_species)
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        mass_fractions = self.getMassFractionsFromVolumeFractions(volume_fractions)
        self.print_fun("{:>18}{:>18}{:>18}{:>18}".format("Species","Volume Fraction","Mole Fraction","Mass Fraction"))
        self.print_fun("{}".format("-"*72))
        for vof,mof,maf,sp in zip(volume_fractions,mole_fractions,mass_fractions,self.species):
            if(sp.name in self.active_species_names):
                self.print_fun("{:>18}{:>18.5g}{:>18.5g}{:>18.5g}".format(sp.name,vof,mof,maf))

    def calcVaporPressure(self,temperature):   # calculate and return vapor pressure from parameters Antoine and Temperature
        vapor_pressure=numpy.power(10.0,self.spA-self.spB/(self.spC+temperature))
        return vapor_pressure

    def calcLiquidVaporBalance(self, temperature, inv_pressure, x0, alpha):
        # equation to be zeroed to close the mole balance
        # (takes temperature, pressure, molar composition liquid, fraction that vaporized)
        K = self.calcVaporPressure(temperature)*inv_pressure - 1.0
        factor = K/(1.0+alpha*K)
        fun=numpy.sum(x0*factor)
        return fun

    def _update_vp_variables(self, temperature, pressure, x0, alpha, bracket=[-10, 1000.0]):
        """
        takes temperature, molar composition liquid, fraction that vaporized and pressure
         and zeroes the balace equation, return Temperature and molar composition liquid and gas
        """
        inv_pressure=1.0/pressure
        try:
            #temperature = optimize.newton(self.calcLiquidVaporBalance, temperature, args=(pressure, x0, alpha), tol=1.00e-06, maxiter=1000)
            temperature = optimize.newton(self.calcLiquidVaporBalance, temperature,
                                          args=(inv_pressure, x0, alpha),
                                          tol=1.00e-02, maxiter=1000)
        except RuntimeError as e:
                try:
                    temperature = optimize.brentq(self.calcLiquidVaporBalance, bracket[0], bracket[1], args=(inv_pressure, x0, alpha))
                except ValueError as e:
                    temperature = optimize.brentq(self.calcLiquidVaporBalance, -10, 1000.0, args=(inv_pressure, x0, alpha))
        except RuntimeError as e:
            return (None, None, None)

        K = self.calcVaporPressure(temperature)*inv_pressure
        xint = x0/(1+alpha*(K-1))
        yint = K*xint
        return temperature, xint, yint

    def write_idt_file(self):
        if(self.logout is None):
            return
        if(len(self.idts)==0):
            return
        with open(self.jobname+'_idts.txt','w') as idt_out_file:
            idt_out_file.write("#{:>18} {:>18} {:>18} {:>18} {:>18}\n".format("Temp (K)",
             "Press (Pa)",  "Phi [-]",  "EGR [-]",   "IDT (s)"))
            for temp, pres, phi, egr, idt in zip(self.temps,self.press,self.phis,self.egrs,self.idts):
                idt_out_file.write(" {:>18.5g} {:>18.5g} {:>18.5g} {:>18.5g} {:>18.5g}\n".format(temp, pres, phi, egr, idt))

    def calcDistillationCurve(self,pressure,mode="Mehl"):
        if(mode=="Mehl"):
           pressure = 760*pressure/101325
           return self.calcDistillationCurveMehl(pressure)
        elif(mode=="REFPROP"):
           return self.calcDistillationCurveREFPROP(pressure)

    def calcDistillationCurveMehl(self,pressure):
        if self.dist_curve_data is not None:
            if(numpy.isclose(pressure,self.dist_curve_data['pressure'])):
                return self.dist_curve_data['return_values']
        volume_fractions = self.getVolumeFractions()
        liquid_mole_fractions = []   # define vector composition liquid (mol)
        vapor_mole_fractions = []   # define vector composition gas   (mol)
        volume_distillate = [] # define vector volume distillate
        distillation_temperatures = []   # define vector Temperature of distillation curve
        mole_fractions = numpy.array(self.getMoleFractionsFromVolumeFractions(volume_fractions))
        T=100.0  #Low starting temperatures can cause root-finding to blow up

        #Bubble Point
        alpha = 0.00
        # evaluates the temperature where the first bubble is formed
        # and the composition of the liquid and of the vapor above the liquid
        (T, x, y) = self._update_vp_variables(T, pressure, mole_fractions, alpha)
        if T is None:
            return (None, None, None, None)

        # Distillation curve obtained using equation for ideal solutions applying
        # Raoult law and a series of molar balance equation on ideal stages (flash separation)
        mole_V = 0.0
        mole_L = 1.0
        Vol_0 = self.getMolarVolumeFromMoleFractions(mole_fractions)
        Vol_1 = self.getMolarVolumeFromMoleFractions(x)
        #calculates the total volume of liquid that has evaporated
        recovered = (Vol_0-mole_L*Vol_1)/Vol_0*100

        volume_distillate.append(recovered)
        distillation_temperatures.append(T)
        mole_fractions_liq_dist = [x]
        mole_fractions_gas_dist = [y]

        #distillation increment
        alpha = self.mehl_distillation_alpha
        while (recovered < 99):
            (T, x, y) = self._update_vp_variables(T, pressure, x, alpha, [T-1,T+5])
            #(T, x, y) = self._update_vp_variables(T, pressure, x, alpha)
            if T is None:
                return (None, None, None, None)

            mole_V = mole_V+mole_L*alpha
            mole_L = 1-mole_V
            # ....and recalculates the total volume of liquid that has evaporated
            Vol_1 = self.getMolarVolumeFromMoleFractions(x)
            recovered = (Vol_0-mole_L*Vol_1)/Vol_0*100

            volume_distillate.append(recovered)
            distillation_temperatures.append(T)
            mole_fractions_liq_dist.append(x)
            mole_fractions_gas_dist.append(y)

        vol_dist = numpy.array(volume_distillate)
        T_dist = numpy.array(distillation_temperatures)
        mole_fractions_liq_dist = numpy.array(mole_fractions_liq_dist)
        mole_fractions_gas_dist = numpy.array(mole_fractions_gas_dist)

        self.dist_curve_data = {}
        self.dist_curve_data['pressure'] = pressure
        self.dist_curve_data['return_values'] = (T_dist, vol_dist, mole_fractions_liq_dist, mole_fractions_gas_dist)
        return self.dist_curve_data['return_values']

    def calcDistillationTemps(self,Volumes=[10,50,90],pressure=101325,offset=4.9,mode="Mehl"):
        T_dist, Vol_dist, mole_fracs_liq_dist, mole_fracs_gas_dist = self.calcDistillationCurve(pressure,mode=mode)
        Temps=[]
        if T_dist is not None:
            Vol_dist -= offset
            for tvr in Volumes:
                idx = 0
                while (Vol_dist[idx+1] < tvr
                    and idx < len(Vol_dist) - 2) :
                     idx += 1
                calculated_temp =( (T_dist[idx+1] - T_dist[idx]) /
                                   (Vol_dist[idx+1] - Vol_dist[idx]) ) \
                                   * (tvr-Vol_dist[idx]) + T_dist[idx]
                Temps.append(calculated_temp)
        return Temps


    def calcDistillationCurveREFPROP(self, pressure=101325):
        if self.dist_curve_data is not None:
            if(numpy.isclose(pressure,self.dist_curve_data['pressure'])):
                return self.dist_curve_data['return_values']

        pressure /= 1000.0  #  Pa -> kPa
        from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
        if 'RPPREFIX' not in os.environ:
            os.environ['RPPREFIX']="/usr/apps/advcomb/opt/REFPROP"
        RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
        RP.SETPATHdll(os.environ['RPPREFIX'])

        mech_to_refprop = {
            "NC16H34" : "C16",
            "NC18H38" : "C18",
            "NC20H42" : "C20",
            "2MHPD"   : "2MC17",
            "NBCH"   : "C4CC6",
            "TIPCX"   : "135TPCC6",
            "PHP"   : "TDHP",
            "TIPB"   : "135TIPBZ",
            "TETRA"   : "TETRALIN",
            "A2CH3"   : "1MNAPHTH",
            "T124MBZ"   : "124MBEN",
            #"T124MBZ"   : "124TRIMETHYLBENZENE",
            "T135MBZ"   : "135TRIMETHYLBENZENE",
            "DECALIN"   : "TDEC",
            "HMN"   : "ISOC16",
            "CPT"   : "CYCLOPEN",
            "CHX"   : "CYCLOHEX",
            "C6H5CH3": "TOLUENE",
            "O-XYL"  : "OXYLENE",
            "IC8"    : "IOCTANE",
            "NC7H16" : "HEPTANE",
            "IC5H12" : "IPENTANE",
            "NC5H12" : "PENTANE",
            "C6H14-2": "IHEXANE",
            "C7H16-2": "ISOC7",
            "C6H12-1": "1HEXENE",
            "C5H10-1": "1PENTENE",
            "C2H5OH" : "ETHANOL",
            "C6H6"   : "BENZENE",
            "NEC6"   : "22DIMETHYLBUTANE",
        }

        mf_dict = self.getMoleFractionsDict()
        for sp in mf_dict:
            if sp not in mech_to_refprop:
                print(f'REFPROP Distillation Error: Unknown species {sp} in composition.')
                return (None, None, None, None)

        fluids = " * ".join([mech_to_refprop[sp] for sp in mf_dict])
        o = RP.SETFLUIDSdll(fluids)
        if(o != 0):
            msg = RP.ERRMSGdll(o)
            if('warning' not in msg):
                print(msg)

        nmoles = 1.0
        vv = 0.0
        iph = 1
        mole_fractions = [mf_dict[sp] for sp in mf_dict]
        x = list(mole_fractions)
        n = list(mole_fractions)
        v = None
        nc = len(mole_fractions)

        vol_dist= []
        T_dist= []
        mole_fractions_liq_dist = []
        mole_fractions_vap_dist = []
        for j in range(500):
            o = RP.SATPdll(pressure,x,iph)
            if(o.ierr != 0):
                print(o.herr)
                return (None, None, None, None)

            #  get initial volume of liquid at nmoles=1
            if j == 0:
                v=1.0/o.Dl
            temperature = o.T
            xliq = o.x
            xvap = o.y

            #  assume a constant amount of moles are lost each step
            #  this is only an approximation and may not reflect reality in the actual experiment
            #  you could alternatively assume a constant mass is lost, a constant volume, or even provide
            #  your own personalized rate of change just make sure you have enough steps so it isn't a function of step size
            #  we found 0.01 seemed reasonable
            nlost = 0.01 # fixed step size
            nmoles -= nlost
            #  reduce each mole number by the amount lost times its mole faction in the vapor
            nsum=0
            for i in range(nc):
                n[i] -= nlost*xvap[i]
                if(n[i] > 0): nsum += n[i]
            for i in range(nc):
                x[i] = max(n[i]/nsum, 0.0)
            #  assume vapor fraction condenses outside of the system;
            #  find liquid density from the lost vapor that condensed outside the system
            o = RP.SATPdll(pressure,xvap,iph)
            if(o.ierr != 0):
                print(o.herr)
                #return (None, None, None, None)

            #  add the volume lost to the total liquid recovered
            vv += nlost/o.Dl
            vol_dist.append(vv/v)          # volume fraction left in the kettle
            T_dist.append(temperature)   # temperature in the kettle 
            mole_fractions_liq_dist.append(xliq)
            mole_fractions_vap_dist.append(xvap)

            if(vv/v > 0.95):
                break

        T_dist = numpy.array(T_dist)-273.15
        vol_dist = numpy.array(vol_dist)*100
        mole_fractions_liq_dist = numpy.array(mole_fractions_liq_dist)
        mole_fractions_vap_dist = numpy.array(mole_fractions_vap_dist)
        self.dist_curve_data = {}
        self.dist_curve_data['pressure'] = pressure
        self.dist_curve_data['return_values'] = (T_dist, vol_dist, mole_fractions_liq_dist, mole_fractions_vap_dist)
        return self.dist_curve_data['return_values']

    def calcCarbonNumberDistribution(self, basis="mole"):
        volume_fractions = self.getVolumeFractions()
        cn_species = self.getCarbonNumbers()
        max_carbon = numpy.max(cn_species)
        carbon_number_dist = numpy.zeros( (max_carbon,))
        fractions = volume_fractions
        if basis == "mole":
            fractions = numpy.array(self.getMoleFractionsFromVolumeFractions(volume_fractions))
        elif basis == "mass":
            fractions = numpy.array(self.getMassFractionsFromVolumeFractions(volume_fractions))
        for f,cn in zip(fractions,cn_species):
            carbon_number_dist[cn-1] += f
        return carbon_number_dist

    def calcCarbonType(self):
        volume_fractions = self.getVolumeFractions()
        ct_species = self.getCarbonTypes()
        carbon_type_mix = numpy.zeros_like(ct_species[0]) # define vector carbon types
        mole_fractions = numpy.array(self.getMoleFractionsFromVolumeFractions(volume_fractions))
        for mf,ct in zip(mole_fractions,ct_species):
            carbon_type_mix += mf*ct
        # Normalize the mole fraction dict
        ct_sum = numpy.sum(carbon_type_mix)
        if ct_sum < 1.0e-300: # don't try to normalize the mole fractions
            ct_sum = 1.0      # if close to zero or negative
        carbon_type_mix = carbon_type_mix/ct_sum
        return carbon_type_mix

    def calcAtomType(self, normalize=True):
        volume_fractions = self.getVolumeFractions()
        at_species = self.getAtomTypes()
        atom_type_mix = numpy.zeros_like(at_species[0]) # define vector carbon types
        mole_fractions = numpy.array(self.getMoleFractionsFromVolumeFractions(volume_fractions))
        for mf,at in zip(mole_fractions,at_species):
            atom_type_mix += mf*at
        # Normalize the mole fraction dict
        if normalize:
            at_sum = numpy.sum(atom_type_mix)
            if at_sum < 1.0e-300: # don't try to normalize the mole fractions
                at_sum = 1.0      # if close to zero or negative
            atom_type_mix = atom_type_mix/at_sum
        return atom_type_mix

    #TODO: Refactor mole fraction averaged calcs (CT, AT, JG)
    def calcJobackGroups(self, normalize=True):
        volume_fractions = self.getVolumeFractions()
        jg_species = self.getJobackGroups()
        jg_mix = numpy.zeros_like(jg_species[0])
        mole_fractions = numpy.array(self.getMoleFractionsFromVolumeFractions(volume_fractions))
        for mf,jg in zip(mole_fractions,jg_species):
            jg_mix += mf*jg
        # Normalize the mole fraction dict
        if normalize:
            jg_sum = numpy.sum(jg_mix)
            if jg_sum < 1.0e-300: # don't try to normalize the mole fractions
                jg_sum = 1.0      # if close to zero or negative
            jg_mix = jg_mix/jg_sum
        return jg_mix

    def calcYSI(self,mode = "mass"):
        assert(mode == "mass" or mode == "mole" or mode == "CT")
        # Das et al.:  https://doi.org/10.1016/j.fuel.2017.01.099
        volume_fractions = self.getVolumeFractions()
        mass_fractions = self.getMassFractionsFromVolumeFractions(volume_fractions)
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        if mode == "mass":
            YSI = numpy.sum([mf*sp.YSI for mf, sp in zip(mass_fractions, self.species)])
        elif mode == "mole":
            YSI = numpy.sum([mf*sp.YSI for mf, sp in zip(mole_fractions, self.species)])
        else: #mode == "CT"
            YSI_coeffs = [-4.49, 1.52, 2.11, 2.71, 19.02, 13.07, 20.61, 70.23, 85.55, 112.09, 23.7]
            YSI_sp = numpy.zeros(len(self.species))
            for i,sp in enumerate(self.species):
                YSI_sp[i] = numpy.sum([ct*coeff for ct, coeff in zip(sp.carbon_type, YSI_coeffs)])
            YSI = numpy.sum([mf*ys for mf, ys in zip(mass_fractions, YSI_sp)])
            #YSI = numpy.sum([mf*ys for mf, ys in zip(mole_fractions, YSI_sp)])
        return YSI

    def calcMW(self):
        volume_fractions = self.getVolumeFractions()
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        MW = numpy.sum([mf*sp.MM for mf, sp in zip(mole_fractions,self.species)])
        return MW

    def calcPMI(self):   # calculate and return vapor pressure from parameters Antoine and Temperature
        vapor_pressures = self.calcVaporPressure(170)
        dbes = [sp.calcDBE() for sp in self.species]
        volume_fractions = self.getVolumeFractions()
        mass_fractions = self.getMassFractionsFromVolumeFractions(volume_fractions)
        pmi = 0
        for vp, dbe, mf in zip(vapor_pressures, dbes, mass_fractions):
            if(mf > 0):
                vp *= 1.01325/760.0 #from mmHg to kPa
                pmi += (dbe+1)/vp*mf
        return pmi

    def _get_zerork_exe(self):
        locations = [ os.getenv("ZERORK_EXE",
                                default=os.path.join(ZERORK_ROOT,'bin','constVolumeWSR.x')),
                           './constVolumeWSR.x']
        for loc in locations:
            if os.path.isfile(loc):
                return loc
        #TODO: Document/inform user
        return 'constVolumeWSR.x'

    def _get_zerork_mpi_exe(self):
        locations = [ os.getenv("ZERORK_MPI_EXE",
                                default=os.path.join(ZERORK_ROOT,'bin','constVolumeWSR_mpi.x')),
                           './constVolumeWSR.x']
        for loc in locations:
            if os.path.isfile(loc):
                return loc
        #TODO: Document/inform user
        return 'constVolumeWSR_mpi.x'

    def _make_oxidizer_mole_fracs(self,params):
        if "oxidizer_mole_fractions" not in params:
            oxidizer_mole_fractions = {"O2":0.21, "N2":0.79}  #TODO: Fix for lower case mechs
        else:
            oxidizer_mole_fractions = params["oxidizer_mole_fractions"]
        return oxidizer_mole_fractions

    def _make_trace_mole_fracs(self,params):
        if "trace_mole_fractions" not in params:
            trace_mole_fractions = {}
        else:
            trace_mole_fractions = params["trace_mole_fractions"]
        return trace_mole_fractions

    def _make_fuel_mole_fracs(self,volume_fractions):
        #TODO: Replace with getMoleFractionsDict ?
        assert(len(volume_fractions) == self.num_species)
        #first get mass fractions
        mole_fractions = self.getMoleFractionsFromVolumeFractions(volume_fractions)
        fuel_mole_fractions = {}
        for mf,sp in zip(mole_fractions,self.species):
            if mf > 0:
                fuel_mole_fractions[sp.name] = float(mf)
        return fuel_mole_fractions

    def _reset_idt_storage(self):
        self.temps=[]
        self.press=[]
        self.phis=[]
        self.egrs=[]
        self.idts=[]
        self.calc_species = []
        self.species_init_mole_fracs = []

    def _optimize_func(self,z,params, force_print=False):
        self.num_evals += 1
        fn_ret = self._set_volume_fractions_from_z(z,params)
        if(fn_ret < 0):
            return 1000.0
        fn_ret = self._idt_func(z,**params)
        if(fn_ret < 0):
            return 1000.0
        else :
            ret_val = self._evaluate_targets(**params)
            if ret_val < self.best_val:
                self.best_val = ret_val
                self.vfs_best.append( (self.getVolumeFractions(), ret_val) )
            best_ratio = ret_val/self.printed_best
            if(best_ratio < 1-self.print_ratio or
               (ret_val == self.best_val and self.num_evals - self.last_print > self.print_interval) or
               (force_print and best_ratio <= 1)):
                self.printed_best = self.best_val
                self.last_print = self.num_evals
                #Print details on best values
                self.print_fun("{}".format("-"*72))
                self.print_fun(f"Status Update (nsteps={self.num_evals}, objective ratio={best_ratio:0.4g})")
                self.print_fun("{}".format("-"*72))
                self.printFractions(self.getVolumeFractions())
                self.print_fun("{}".format("-"*72))
                self._print_targets(**params)
                self.print_fun(("{:>18}"*3+"{:>18.7g}").format("Total Objective","--","--",ret_val))
                self.print_fun("{}".format("-"*72))
                self.print_fun("")

                #Write out any files
                self.write_idt_file()
                for t in params['targets']:
                    t.write_file(self.jobname)

                with open("{}.p".format(self.jobname),'wb') as pfile:
                    dill.dump( [self, params["targets"]], pfile )
            return ret_val

    def _set_volume_fractions_from_z(self,z,params):
        opt_params = params['opt_params']
        vf_loc = numpy.zeros((self.num_species,))
        vf_loc[opt_params] = self._z_to_vf(z, params)

        self.setVolumeFractionsArray(vf_loc)
        return 1

    def _idt_func(self,z,**params):
        self._reset_idt_storage()

        for sw in params['idt_sweeps']:
            temps = sw['temps']
            press = sw['press']
            phis  = sw['phis']
            egrs  = sw['egrs']
            fn_ret = self._zerork_func(temps,press,phis,egrs,**params)
            if(fn_ret < 0): return fn_ret
        return 1

    def _zerork_func(self,temps,press,phis,egrs,**params):
        import tempfile, shutil
        if(self.mf is None or self.tf is None):
            print("Can't run ZeroRK without mech_file and therm_file")
            sys.exit()

        vf_loc = self.getVolumeFractions()

        tmpdir = tempfile.mkdtemp(dir="./")
        zerork_out_file=open(os.path.join(tmpdir,self.jobname+'.out'),'a')

        zerork_out_file.write('!!! Running ZeroRK with volume fractions={}\n'.format(vf_loc))

        zrk_yaml = dict()
        zrk_yaml['mechFile'] = self.mf
        zrk_yaml['thermFile'] = self.tf
        zrk_yaml['logFile'] = os.path.join(tmpdir,self.jobname+'.cklog')
        zrk_yaml['idtFile'] = os.path.join(tmpdir,self.jobname+'.dat')
        zrk_yaml['thistFile'] =  "/dev/null"  #used to be non-null
        zrk_yaml['stop_time'] = params["stopping_time"]
        zrk_yaml['print_time'] = params["stopping_time"]
        zrk_yaml['temperature_deltas'] = [params["delta_temp_ignition"]]
        zrk_yaml['initial_temperatures'] = [float(x) for x in temps]
        zrk_yaml['initial_pressures'] = [float(x) for x in press]
        zrk_yaml['initial_phis'] = [float(x) for x in phis]
        zrk_yaml['initial_egrs'] = [float(x) for x in egrs]
        zrk_yaml['oxidizer_mole_fracs'] = self._make_oxidizer_mole_fracs(params)
        zrk_yaml['trace_mole_fracs'] = self._make_trace_mole_fracs(params)
        zrk_yaml['fuel_mole_fracs'] = self._make_fuel_mole_fracs(vf_loc)
        zrk_yaml['absolute_tolerance'] = params['abs_tol']
        zrk_yaml['relative_tolerance'] = params['rel_tol']
        zrk_yaml['preconditioner_thresholds'] = [2.048e-3]
        zrk_yaml['eps_lin'] = 0.05
        zrk_yaml['max_internal_dt'] = 0.05
        zrk_yaml['continue_after_ignition'] = 0

        zerork_infile_name = os.path.join(tmpdir,self.jobname +'.yml')
        with open(zerork_infile_name,'w') as yfile:
            yaml=YAML(typ="safe")
            yaml.dump(zrk_yaml, yfile)
        zerork_out=''
        try:
            if('mpi_procs' in params and params['mpi_procs'] > 1 and self.zerork_mpi_exe):
                np=str(params['mpi_procs'])
                mpi_cmd = params.get('mpi_cmd','srun -n')
                if(mpi_cmd == 'mpirun') : mpi_cmd += ' -np'
                if(mpi_cmd == 'srun') : mpi_cmd += ' -n'
                cmd_list = params['mpi_cmd'].split() + [np,self.zerork_mpi_exe,zerork_infile_name]
                zerork_out=subprocess.check_output(cmd_list, stderr=subprocess.STDOUT,
                                                   universal_newlines=True).split('\n')
            else:
                zerork_out=subprocess.check_output([self.zerork_exe,zerork_infile_name],
                                                   stderr=subprocess.STDOUT,universal_newlines=True).split('\n')
        except subprocess.CalledProcessError as e:
            zerork_out_file.write('!!! Warning: ZeroRK exited with non-zero output ({}).\n'.format(e.returncode))
            zerork_out=e.output.split('\n')

        for line in zerork_out:
            zerork_out_file.write(line+'\n')

        start_data=False
        try:
            with open(os.path.join(tmpdir,self.jobname+'.dat'),'r') as datfile:
                for line in datfile:
                    if len(line) <= 1:
                        if start_data: break #done with first block break out
                        start_data = False
                        continue
                    if line[0] != '#':
                        start_data = True
                    if "run id" in line:
                        tokens = line.split()
                        tmp_list = []
                        for i,tok in enumerate(tokens):
                            if tok == "mlfrc":
                                tmp_list.append(tokens[i+1])
                        if len(tmp_list) > 0:
                            self.calc_species.append(tmp_list)
                    if start_data:
                        vals = list(map(float,line.split()))
                        self.temps.append(vals[1])
                        self.press.append(vals[2])
                        self.phis.append(vals[3])
                        self.egrs.append(vals[4])
                        self.idts.append(vals[7])
                        self.species_init_mole_fracs.append(vals[10:])
                        assert(len(self.species_init_mole_fracs[-1]) == len(self.calc_species[-1]))

        except IOError:
            print("No data file from ZeroRK, ZeroRK output was:")
            for line in zerork_out:
                print("\t", line)
            raise

        finally:
            zerork_out_file.close()
            #Clean up
            try:
                shutil.rmtree(tmpdir)
            except OSError:
                try:
                    time.sleep(0.5)
                    shutil.rmtree(tmpdir)
                except OSError as e:
                    print(f"WARNING: Couldn't remove tmpdir: {tmpdir}")
                    print(e)

        return 1 #success

    def _evaluate_targets(self, **params):
        targets = params['targets']
        self.dist_curve_data = None

        #Calculate errors
        total_err = 0.0
        for t in targets:
            total_err += t(self)**2
        total_err = numpy.sqrt(total_err)
        return total_err

    def _print_targets(self, **params):
        targets = params['targets']
        self.print_fun(("{:>18}"*4).format("Target Name","Target Value","Current Value","Objective"))
        self.print_fun("{}".format("-"*72))
        for t in targets:
            self.print_fun(t.printout())

    @staticmethod
    def AKI_2011_func(log10idt825):
        x = log10idt825
        y = 67.95*x**3 + 422.17*x**2 + 903.17*x + 753.93
        AKI = y
        return AKI

    @staticmethod
    def AKI_func(log10idt825):
        x = log10idt825
        #y = -33.66*x**4 - 259.05*x**3 - 750.37*x**2 - 942.93*x - 326.63
        #AKI=IF(TAU825<-1.5,-33.66* TAU825^4 - 259* TAU825^3 - 750.4* TAU825^2 - 942.9* TAU825- 326.6,13.841* TAU825 + 123.8)
        if(x < -1.5):
            y = -33.66* x**4 - 259* x**3 - 750.4* x**2 - 942.9* x- 326.6
        else:
            y = 13.841*x + 123.8
        AKI = y
        return AKI

    @staticmethod
    def RON_func(log10idt775):
        x = log10idt775
        #RON =IF LOGTAU(775)<-1.7, -37.493* LOGTAU(775)^4-306.63* LOGTAU(775)^3-954.9* LOGTAU(775)^2-1316.6* LOGTAU(775)-567.45
        #ELSE 6.0299* LOGTAU(775)+115.26
        if(x <= -1.7):
            y = -37.493*log10idt775**4-306.63*log10idt775**3 - 954.9*log10idt775**2-1316.6*log10idt775-567.45
        else:
            #y = 6.0299*log10idt775 + 115.26
            y = 6.0299*(log10idt775 + 1.7) + SurrogateMixture.RON_func(-1.7)
        RON = y
        return RON

    def log10idt825(self):
        #Get IDT at 825 K, 25 bar, phi=1.0, egr=0.0
        log10idt825 = numpy.Inf
        for temp,p,phi,egr,idt in zip(self.temps,self.press,self.phis,self.egrs,self.idts):
            if(numpy.abs(temp-825.0) < 1.0 and numpy.abs(p-25.0*101325.0) < 1.0
                and numpy.abs(phi-1.0) < 0.001 and numpy.abs(egr) < 0.001):
                log10idt825 = numpy.log10(idt)
                break
        assert(numpy.isfinite(log10idt825))
        return log10idt825

    def log10idt775(self):
        #Get IDT at 775 K, 25 bar, phi=1.0, egr=0.0
        log10idt775 = numpy.Inf
        for temp,p,phi,egr, idt in zip(self.temps,self.press,self.phis,self.egrs,self.idts):
            if(numpy.abs(temp-775.0) < 1.0 and numpy.abs(p-25.0*101325.0) < 1.0
                and numpy.abs(phi-1.0) < 0.001 and numpy.abs(egr) < 0.001):
                log10idt775 = numpy.log10(idt)
                break
        assert(numpy.isfinite(log10idt775))
        return log10idt775

    def AKI_Correlation_2011(self):
        return self.AKI_2011_func(self.log10idt825())

    def AKI_Correlation(self):
        return self.AKI_func(self.log10idt825())

    def RON_Correlation(self):
        return self.RON_func(self.log10idt775())

    @staticmethod
    def SENS_2011_func(min_slope):
        x = min_slope
        y = -0.60*x**2 + 2.64*x + 8.86
        SENS = y
        return SENS

    @staticmethod
    def SENS_V1_func(min_slope):
        x = min_slope
        #y = 11.28/(1+numpy.exp(-(3.36*x+2.11)))
        y = 11.2841533651579/(1+numpy.exp(-(x*3.36579590239894+2.11347810796132)))
        SENS = y
        return SENS

    @staticmethod
    def SENS_ETOH_func(min_slope):
        x = min_slope
        y = (8.5+1.3*x)/(1+numpy.exp(-(x*3.36579590239894+2.11347810796132)))
        SENS = y
        return SENS

    @staticmethod
    def SENS_V3_func(min_slope):
        #OS= 0.0025*MINSLOPE^4 - 0.0128* MINSLOPE ^3 - 0.3891* MINSLOPE ^2 + 3.1433* MINSLOPE + 9.2443
        x = min_slope
        y = 0.0025*min_slope**4 - 0.0128*min_slope**3 - 0.3891*min_slope**2 + 3.1433*min_slope + 9.2443
        SENS = y
        return SENS

    def min_slope(self):
        sens_vals = []
        for temp,p,phi,egr, idt in zip(self.temps,self.press,self.phis,self.egrs,self.idts):
            if(numpy.abs(p-25.0*101325.0) < 1.0 and numpy.abs(phi-1.0) < 0.001 and numpy.abs(egr) < 0.001):
                sens_vals.append((temp,idt))

        sens_vals = list(set(sens_vals))
        sens_vals.sort()  # list of unique temps and idts; sorted by temp
        assert( len(sens_vals) > 2 )
        t_min = sens_vals[0][0]
        t_max = sens_vals[-1][0]
        assert( t_min <= 650.0)
        assert( t_max >= 1000.0)
        invTemps  = [1000.0/sv[0] for sv in sens_vals]
        log10idts = [numpy.log10(sv[1]) for sv in sens_vals]

        #Calc minimum slope of idt
        min_slope = numpy.Inf
        for idx in range(1,len(invTemps)-1):
            slope = 0.5*(
             (log10idts[idx+1] - log10idts[idx]  )/(invTemps[idx+1]-invTemps[idx]  ) +
             (log10idts[idx]   - log10idts[idx-1])/(invTemps[idx]  -invTemps[idx-1])
            )
            if slope < min_slope:
                min_slope = slope
        assert(numpy.isfinite(min_slope))
        return min_slope

    def SENS_Correlation_2011(self):
        return self.SENS_V1_func(self.min_slope())

    def SENS_Correlation_V1(self):
        return self.SENS_V1_func(self.min_slope())

    def SENS_CorrelationETOH(self):
        return self.SENS_ETOH_func(self.min_slope())

    def SENS_Correlation_V3(self):
        return self.SENS_V3_func(self.min_slope())

    def _z_to_vf_mult(self, z, params):
        bulk_idx = params["bulk_idx"]
        scale = params["param_scale"]

        vfs = numpy.zeros((len(z)+1,))
        z = numpy.clip(z, 1.0e-8*params["param_scale"], 8*params["param_scale"])
        center = 1.0/(len(vfs))
        vfs = [max(1.0e-8,center*(1.0+scale*(zi-1.0))) for zi in z]
        vf_sum = numpy.sum(vfs)
        if(vf_sum < (1.0-1.0e-8)):
            vfs.insert(bulk_idx,1 - numpy.sum(vfs))
        else:
            vfs.insert(bulk_idx,1.0e-8)
            vfs /= (vf_sum+1.0e-8)
        return vfs

    def _vf_to_z_mult(self, vfs, params):
        bulk_idx = params["bulk_idx"]
        scale = params["param_scale"]
        z = numpy.array(vfs)
        z = numpy.clip(numpy.delete(z, bulk_idx), 1.0e-30, None)
        return scale*len(vfs)*z

    def _z_to_vf_logit(self, z, params):
        bulk_idx = params["bulk_idx"]
        scale = params["param_scale"]
        vfs = numpy.zeros((len(z)+1,))
        z = numpy.clip(z, -18.42/params["param_scale"], 3.5/params["param_scale"])
        center = 1.0/(len(vfs))
        vfs = [max(1.0e-8,1/(1+numpy.exp(-(scale*zi+numpy.log(center/(1-center)))))) for zi in z]
        vf_sum = numpy.sum(vfs)
        if(vf_sum < (1.0-1.0e-8)):
            vfs.insert(bulk_idx,1 - numpy.sum(vfs))
        else:
            vfs.insert(bulk_idx,1.0e-8)
            vfs /= (vf_sum+1.0e-8)
        return vfs

    def _vf_to_z_logit(self, vfs, params):
        bulk_idx = params["bulk_idx"]
        scale = params["param_scale"]
        z = numpy.array(vfs)
        z = numpy.clip(numpy.delete(z, bulk_idx), 1.0e-30, None)
        return (numpy.log(z/(1-z))+numpy.log(len(vfs)-1))/scale

    def _z_to_vf_alr(self, z, params):
        bulk_idx = params["bulk_idx"]
        z = numpy.clip(z, -80, 80)
        bulk_val = 1/(numpy.sum(numpy.exp(z))+1)
        vfs = list(bulk_val*numpy.exp(z))
        vfs.insert(bulk_idx,bulk_val)
        return vfs

    def _vf_to_z_alr(self, vfs, params):
        bulk_idx = params["bulk_idx"]
        denom = vfs[bulk_idx]
        z = numpy.array(vfs)
        z = numpy.clip(numpy.delete(z, bulk_idx), 1.0e-30, None)
        return numpy.log(z/denom)

    def setTransform(self, mode):
        assert(mode in ["mult", "logit", "alr"])
        if mode == "mult":
          self._vf_to_z = self._vf_to_z_mult
          self._z_to_vf = self._z_to_vf_mult
          self.transform_mode = "mult"
        if mode == "logit":
          self._vf_to_z = self._vf_to_z_logit
          self._z_to_vf = self._z_to_vf_logit
          self.transform_mode = "logit"
        if mode == "alr":
          self._vf_to_z = self._vf_to_z_alr
          self._z_to_vf = self._z_to_vf_alr
          self.transform_mode = "alr"

    def _merge_sweeps(self,params):
        duplicate_points = []
        for sw in params['idt_sweeps']:
            if(sw in duplicate_points): continue
            if( len(sw['temps']) == 1 and
                len(sw['press']) == 1 and
                len(sw['phis']) == 1 and
                len(sw['egrs']) == 1):
                t = sw['temps'][0]
                p = sw['press'][0]
                f = sw['phis'][0]
                e = sw['egrs'][0]
                for sw2 in params['idt_sweeps']:
                    if sw is sw2: continue
                    if (t in sw2['temps'] and
                        p in sw2['press'] and
                        f in sw2['phis'] and
                        e in sw2['egrs']):
                        duplicate_points.append(sw)

        for dp in duplicate_points:
            params['idt_sweeps'].remove(dp)

        #Merge temp sweeps at identicial p,f,e
        remove_idxs = []
        for i,sw1 in enumerate(params['idt_sweeps']):
            if( len(sw1['press']) == 1 and
                len(sw1['phis']) == 1 and
                len(sw1['egrs']) == 1):
                p1 = sw1['press'][0]
                f1 = sw1['phis'][0]
                e1 = sw1['egrs'][0]
                for j,sw2 in enumerate(params['idt_sweeps'][i+1:]):
                    if( len(sw2['press']) == 1 and
                        len(sw2['phis']) == 1 and
                        len(sw2['egrs']) == 1):
                        p2 = sw2['press'][0]
                        f2 = sw2['phis'][0]
                        e2 = sw2['egrs'][0]
                        if(p1==p2 and f1==f1 and e1==e2):
                            params['idt_sweeps'][i]['temps'] = list(set(sw1['temps']+sw2['temps']))
                            remove_idxs.append(j+i+1)

        remove_idxs.reverse()
        for idx in remove_idxs:
            params['idt_sweeps'].pop(idx)

    def _setup_eval(self, vf_guess, targets, jobname, stdout, logout, **kwargs):
        self.best_val = numpy.inf
        self.printed_best = numpy.inf
        self.num_evals = 0
        self.last_print = 0
        self.vfs_best = []
        self.dist_curve_data = None
        self.stdout = stdout
        self.logout = logout

        if jobname is None or len(jobname)==0:
            jobname = "surrogate_optimizer"
        self.jobname=jobname

        if self.logout and self.logfile is None:
            self.logfile = open(self.jobname+".log",'w')

        self.setVolumeFractionsDict(vf_guess)
        opt_species = list(vf_guess.keys())
        bulk_idx=-1
        bulk_specie = None
        max_vf = 0.0
        for idx, sp in enumerate(opt_species):
            if vf_guess[sp] > max_vf:
                max_vf = vf_guess[sp]
                bulk_specie = sp
                bulk_idx = idx
        assert(bulk_specie!="")
        assert(bulk_idx!=-1)

        opt_params = self.getSpeciesIndices(opt_species)
        bulk_idx = opt_params.index(self.getSpeciesIndices([bulk_specie])[0])
        volume_fractions = numpy.array(self.getVolumeFractions())

        param_scale = 1.0

        params = {}
        params['opt_params'] = opt_params
        params['volume_fractions'] = volume_fractions
        params['param_scale'] = param_scale
        params['bulk_idx'] = bulk_idx
        params['targets'] = targets
        params['stopping_time'] = 1000.0
        params['delta_temp_ignition'] = 400.0
        params['opt_method'] = 'Nelder-Mead'
        params['abs_tol'] = 1.0e-20
        params['rel_tol'] = 1.0e-08
        params['mpi_procs'] = 1
        params['mpi_cmd'] = 'srun -n'
        params.update(kwargs)
        # Sub-dicts update with defaults:
        params['opt_kwargs'] = {'tol':1.0e-3}
        params['opt_options'] = {}
        if 'opt_options' in kwargs:
            params['opt_options'].update(kwargs['opt_options'])
        if 'opt_kwargs' in kwargs:
            params['opt_kwargs'].update(kwargs['opt_kwargs'])

        params['idt_sweeps'] = []
        if 'temps' in params:
            sw = {}
            sw['temps'] = params['temps']
            sw['press'] = params.get('press',[25.0*101325.0])
            sw['phis'] = params.get('phis',[1.0])
            sw['egrs'] = params.get('egrs',[0.0])
            params['idt_sweeps'].append(sw)
        #remove sweep arguments from params
        params.pop('temps', None)
        params.pop('press', None)
        params.pop('phis', None)
        params.pop('egrs', None)
        for target in targets:
            sweeps = target.get_idt_sweeps()
            if sweeps:
                params['idt_sweeps'].extend(sweeps)
        if(len(params['idt_sweeps']) > 1):
            self._merge_sweeps(params)

        z0 = self._vf_to_z(volume_fractions[opt_params], params)
        return (z0,params)

    def _close_eval(self):
        if(self.logfile):
            self.logfile.close()
            self.logfile = None

    def optimize(self, vf_guess, targets, jobname=None, stdout=True, logout=True, **kwargs):
        (guess,params) = self._setup_eval(vf_guess,targets,jobname,stdout,logout, **kwargs)
        opt_options=params['opt_options']
        opt_method=params['opt_method']
        opt_kwargs=params['opt_kwargs']

        opt_result = optimize.minimize(self._optimize_func, guess, args=params, method=opt_method, options=opt_options, **opt_kwargs)
        result = self._optimize_func(opt_result.x, params, force_print=True)
        self._close_eval()
        return result

    def optimize_ea(self, vfs, targets, jobname=None, stdout=True, logout=True, **kwargs):
        (guess,params) = self._setup_eval(vfs,targets,jobname,stdout,logout, **kwargs)
        opt_kwargs=params['opt_kwargs']
        bounds = None
        if(self.transform_mode == "mult"):
            bounds = [ (1.0e-8*params["param_scale"], 8.0*params["param_scale"]) for _ in guess ]
        if(self.transform_mode == "logit"):
            bounds = [ (-18.42/params["param_scale"], 3.5/params["param_scale"]) for _ in guess ]
        if(self.transform_mode == "alr"):
            bounds = [ (-80, 80) for _ in guess ]
        assert(bounds is not None)
        opt_result = optimize.differential_evolution(self._optimize_func, bounds, args=(params,), polish=False, **opt_kwargs)
        result = self._optimize_func(opt_result.x, params, force_print=True)
        self._close_eval()
        return result

    def optimize_bh(self, vfs, targets, jobname=None, stdout=True, logout=True, **kwargs):
        (guess,params) = self._setup_eval(vfs,targets,jobname,stdout,logout, **kwargs)
        opt_options=params['opt_options']
        opt_method=params['opt_method']
        opt_kwargs=params['opt_kwargs']
        bh_kwargs=kwargs.get('bh_kwargs',{})

        minimizer_kwargs = {}
        minimizer_kwargs['method'] = opt_method
        minimizer_kwargs['options'] = opt_options
        minimizer_kwargs['args'] = params
        minimizer_kwargs.update(opt_kwargs)

        start_time = time.perf_counter()
        max_time = kwargs.get('max_time', 1500)
        def callback(x,f,accept):
            elapsed_time = time.perf_counter() - start_time
            if(elapsed_time > max_time):
                return True
            else:
                return False

        bh_kwargs['callback'] = callback
        opt_result = optimize.basinhopping(self._optimize_func, guess, minimizer_kwargs=minimizer_kwargs, **bh_kwargs)
        result = self._optimize_func(opt_result.x, params, force_print=True)
        self._close_eval()
        return result

    def evaluate(self, vf_guess, targets, jobname=None, stdout=True, logout=True, **kwargs):
        (guess,params) = self._setup_eval(vf_guess,targets,jobname,stdout,logout, **kwargs)
        result = self._optimize_func(guess,params)
        self._close_eval()
        return result

    def get_idts(self, vf, jobname=None, stdout=True, logout=True, **kwargs):
        targets=[]
        (guess,params) = self._setup_eval(vf,targets,jobname,stdout,logout, **kwargs)
        result = self._idt_func(guess,**params)
        if( result != 1 ): print("Error returned from idt func!")
        self._close_eval()
        return self.idts


def AbsoluteErrorFn(clip_value = 0.0):
    def target_fn(target_value, value):
        error = max(numpy.abs(value - target_value) - clip_value, 0.0)
        return error
    return target_fn

def RelativeErrorFn(clip_value = 0.0):
    def target_fn(target_value, value):
        error = numpy.abs(target_value - value)
        if target_value != 0.0:
            error /= target_value
        error = max(error - clip_value, 0.0)
        return error
    return target_fn

def PositiveDifferenceErrorFn(clip_value = 0):
    def target_fn(target_value, value):
        return max(target_value - value, clip_value)
    return target_fn

def NegativeDifferenceErrorFn(clip_value = 0):
    def target_fn(target_value, value):
        return max(value - target_value, clip_value)
    return target_fn

def ExponentialDifferenceErrorFn(exp_weight):
    def target_fn(target_value, value):
        return numpy.exp(exp_weight*(target_value - value))
    return target_fn

class TargetBase(object):
    def __init__(self, name, target_value, weight, target_fn=RelativeErrorFn()):
        self.name = name
        self.target_value = target_value
        self.weight = weight
        self.last_value = numpy.nan
        self.last_err = numpy.nan
        self.target_fn=target_fn
    def __call__(self, value):
        self.last_value = value
        self.last_err = self.target_fn(self.target_value, value)*self.weight
        return self.last_err
    def printout(self):
        return "{0:>18s}{1:>18.4g}{2:>18.4g}{3:>18.4g}".format(self.name, self.target_value, self.last_value, self.last_err)
    def write_file(self,jobname):
        pass
    def get_idt_sweeps(self):
        return None

class AKI2011Target(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"AKI (2011)",target,weight,target_fn)
    def __call__(self, mixture):
        computed_AKI = mixture.AKI_Correlation_2011()
        error = TargetBase.__call__(self,computed_AKI)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [825.0]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]

class AKITarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"AKI",target,weight,target_fn)
    def __call__(self, mixture):
        computed_AKI = mixture.AKI_Correlation()
        error = TargetBase.__call__(self,computed_AKI)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [825.0]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]


class RONTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"RON",target,weight,target_fn)
    def __call__(self, mixture):
        computed_RON = mixture.RON_Correlation()
        error = TargetBase.__call__(self,computed_RON)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [775.0]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]


class SENS2011Target(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"SENS (2011)",target,weight,target_fn)
    def __call__(self, mixture):
        computed_SENS = mixture.SENS_Correlation_2011()
        error = TargetBase.__call__(self,computed_SENS)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [650.0 + i*25.0 for i in range(15)]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]


class SENS_V1Target(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"SENS (V1)",target,weight,target_fn)
    def __call__(self, mixture):
        computed_SENS = mixture.SENS_Correlation_V1()
        error = TargetBase.__call__(self,computed_SENS)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [650.0 + i*25.0 for i in range(15)]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]


class SENS_ETOHTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"SENS_ETOH",target,weight,target_fn)
    def __call__(self, mixture):
        computed_SENS = mixture.SENS_CorrelationETOH()
        error = TargetBase.__call__(self,computed_SENS)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [650.0 + i*25.0 for i in range(15)]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]


class SENSTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"SENS",target,weight,target_fn)
    def __call__(self, mixture):
        computed_SENS = mixture.SENS_Correlation_V3()
        error = TargetBase.__call__(self,computed_SENS)
        return error
    def get_idt_sweeps(self):
        sw = {}
        sw['temps'] = [650.0 + i*25.0 for i in range(15)]
        sw['press'] = [25.0*101325.0]
        sw['phis']  = [1.0]
        sw['egrs']  = [0.0]
        return [sw]


class HCTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"HC",target,weight,target_fn)
    def __call__(self, mixture):
        HC = mixture.calcHCRatio()
        error = TargetBase.__call__(self,HC)
        return error


class VolumeFractionTarget(TargetBase):
    def __init__(self, target, weight, kind, target_fn=RelativeErrorFn()):
        self.kind = kind
        TargetBase.__init__(self,kind+"_vf",target,weight,target_fn)
    def __call__(self, mixture):
        vf = mixture.getKindVolumeFraction(self.kind)
        error = TargetBase.__call__(self,vf)
        return error


class DensityTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"Density",target,weight,target_fn)
    def __call__(self, mixture):
        density = mixture.calcDensity()
        error = TargetBase.__call__(self,density)
        return error
    def get_idt_sweeps(self):
        return None


class CetaneNumberTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"Cetane Number",target,weight,target_fn)
    def __call__(self, mixture):
        volume_fractions = mixture.getVolumeFractions()
        species = mixture.species
        cetane_number = 0.0
        for vf, sp in zip(volume_fractions, species):
            cetane_number += vf*sp.DCN
        error = TargetBase.__call__(self,cetane_number)
        return error
    def get_idt_sweeps(self):
        return None

class PMITarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"PMI",target,weight,target_fn)
    def __call__(self, mixture):
        computed_PMI = mixture.calcPMI()
        error = TargetBase.__call__(self,computed_PMI)
        return error
    def get_idt_sweeps(self):
        return None

class YSITarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn(), mode="mass"):
        TargetBase.__init__(self,"YSI",target,weight,target_fn)
        self.mode = mode
    def __call__(self, mixture):
        computed_YSI = mixture.calcYSI(mode=self.mode)
        error = TargetBase.__call__(self,computed_YSI)
        return error
    def get_idt_sweeps(self):
        return None

class MWTarget(TargetBase):
    def __init__(self, target, weight, target_fn=RelativeErrorFn()):
        TargetBase.__init__(self,"MW",target,weight,target_fn)
    def __call__(self, mixture):
        computed_MW = mixture.calcMW()
        error = TargetBase.__call__(self,computed_MW)
        return error
    def get_idt_sweeps(self):
        return None

class CarbonNumberTarget(TargetBase):
    def __init__(self, target_carbons, total_target_fraction, weight, basis="mole", target_fn=RelativeErrorFn()):
        self.target_carbons=numpy.array([target_carbons],dtype=numpy.int32).flatten()
        self.total_target_fraction=numpy.sum(numpy.array([total_target_fraction]).flatten())
        self.name = "C" + ",".join([str(elem) for elem in self.target_carbons])
        self.basis = basis
        TargetBase.__init__(self,self.name,self.total_target_fraction,weight,target_fn)
    def __call__(self,mixture):
        carbon_number_dist = mixture.calcCarbonNumberDistribution(basis=self.basis)
        max_carbon_number = len(carbon_number_dist)
        total_carbon = numpy.sum([carbon_number_dist[idx-1] for idx in self.target_carbons if idx <= max_carbon_number])
        error = TargetBase.__call__(self,total_carbon)
        return error
    def get_idt_sweeps(self):
        return None

class CarbonTypeTarget(object):
    def __init__(self, ct_data_file, weight):
        self.name = "Carbon Type"
        self.weight = weight
        self.target_value = "--"
        self.last_value = "--"
        self.last_err = numpy.nan
        data = numpy.genfromtxt(ct_data_file,delimiter=",")
        self.target_C_type = data[:,0]
        self.target_C_fraction = data[:,1]
    def __call__(self,mixture):
        carbon_type_mix = mixture.calcCarbonType()
        error = 0.0
        for ct_mix,ct_targ in zip(carbon_type_mix,self.target_C_fraction):
            if ct_targ != 0:
                error += abs(ct_mix-ct_targ)/ct_targ
            else:
                error += abs(ct_mix)
        error *= self.weight
        self.last_err = error
        self.carbon_type_mix = carbon_type_mix
        return error
    def printout(self):
        return "{0:>18s}{1:>18s}{2:>18s}{3:>18.4g}".format(self.name, self.target_value, self.last_value, self.last_err)
    def write_file(self,jobname):
        with open(jobname+'_C_Type.txt','w') as C_out_file:
            for i,CT in enumerate(self.carbon_type_mix):
                C_out_file.write("{},{}\n".format(i+1,CT))
    def get_idt_sweeps(self):
        return None


class SingleCarbonTypeTarget(TargetBase):
    def __init__(self, ct_data_file, ct_num, weight, target_fn=AbsoluteErrorFn()):
        name = "Carbon Type #{}".format(ct_num)
        data = numpy.genfromtxt(ct_data_file,delimiter=",")
        self.ct_num = ct_num
        target = data[ct_num-1,1]
        TargetBase.__init__(self,name,target,weight,target_fn)
    def __call__(self,mixture):
        carbon_type_mix = mixture.calcCarbonType()
        value = carbon_type_mix[self.ct_num-1]
        error = TargetBase.__call__(self,value)
        return error

class LimitedVolumeFractionTarget(TargetBase):
    def __init__(self, target, weight, kind, target_fn=NegativeDifferenceErrorFn()):
        name = "{} < {}%".format(kind,target*100)
        self.kind = kind
        TargetBase.__init__(self,name,target,weight,target_fn)
    def __call__(self, mixture):
        vf = mixture.getKindVolumeFraction(self.kind)
        error = TargetBase.__call__(self,vf)
        return error

class LimitedMassFractionTarget(TargetBase):
    def __init__(self, target, weight, kind, target_fn=NegativeDifferenceErrorFn()):
        name = "{} < {}% (mass)".format(kind,target*100)
        self.kind = kind
        TargetBase.__init__(self,name,target,weight,target_fn)
    def __call__(self, mixture):
        vf = mixture.getKindMassFraction(self.kind)
        error = TargetBase.__call__(self,vf)
        return error

class L2PenaltyTarget(TargetBase):
    def __init__(self, weight):
        name = "L2 Penalty"
        parent_target = 0.0
        TargetBase.__init__(self, name, parent_target, weight, AbsoluteErrorFn())
    def __call__(self, mixture):
        l2 = numpy.sum(numpy.square(mixture.getVolumeFractions()))
        error = TargetBase.__call__(self,l2)
        return error

class DistillationTemperatureTarget(TargetBase):
    def __init__(self, target_percent, target_temperature, weight, target_fn=RelativeErrorFn(),mode="Mehl"):
        self.name = "T{}".format(target_percent)
        self.target_percent = target_percent
        self.mode = mode
        TargetBase.__init__(self,self.name,target_temperature,weight,target_fn)
    def __call__(self, mixture):
        T = mixture.calcDistillationTemps(Volumes=[self.target_percent],
                                          offset=0.0,mode=self.mode)[0]
        error = TargetBase.__call__(self,T)
        return error

class DistillationCurveTarget(object):
    def __init__(self, dc_data_file, weight, units='K', pressure=101325, offset=0.0, mode="Mehl"):
        self.name = "Dist. Curve"
        self.weight = weight
        self.pressure = pressure
        self.offset = offset
        self.target_value = "--"
        self.last_value = "--"
        self.last_err = numpy.nan
        data = numpy.genfromtxt(dc_data_file,delimiter=",")
        self.target_volume_recovered = data[:,0]
        self.target_temperatures = data[:,1]
        self.mode=mode

        if units != "C":
            if units == "K":
                self.target_temperatures = [t-273.15 for t in self.target_temperatures]
            elif units == "F":
                self.target_temperatures = [(t-32.0)*5.0/9.0 for t in self.target_temperatures]
            else:
                print("Unsupported units for distillation curve: %s" % units)
                sys.exit()
    def __call__(self, mixture):
        T_dist, Vol_dist, mole_fracs_liq_dist, mole_fracs_gas_dist = mixture.calcDistillationCurve(self.pressure,mode=self.mode)
        if T_dist is not None:
            error = 0.0
            Vol_dist -= self.offset
            for tvr, tt in zip(self.target_volume_recovered, self.target_temperatures):
                idx = 0
                while (Vol_dist[idx+1] < tvr
                    and idx < len(Vol_dist) - 2) :
                     idx += 1
                calculated_temp =( (T_dist[idx+1] - T_dist[idx]) /
                                   (Vol_dist[idx+1] - Vol_dist[idx]) ) \
                                   * (tvr-Vol_dist[idx]) + T_dist[idx]
                error += numpy.abs((tt-calculated_temp)/(tt+273.15)) #Normalize with K
            #Multiply weight and divide by number of points
            error *= self.weight/len(self.target_volume_recovered)
        else:
            error = 100.0
        self.last_err = error
        self.T_dist = T_dist 
        self.Vol_dist = Vol_dist 
        self.mole_fracs_liq_dist = mole_fracs_liq_dist 
        self.mole_fracs_gas_dist = mole_fracs_gas_dist 
        return error
    def printout(self):
        return "{0:>18s}{1:>18s}{2:>18s}{3:>18.4g}".format(self.name, self.target_value, self.last_value, self.last_err)
    def write_file(self,jobname):
        with open(jobname+'_dist_curve.txt','w') as dist_out_file:
            for vr, t in zip(self.Vol_dist,self.T_dist):
                dist_out_file.write("{},  {}\n".format(vr, t))
        with open(jobname+'_dist_curve_mole_fracs.txt','w') as mf_out_file:
            for t,mfs in zip(self.T_dist,self.mole_fracs_gas_dist):
                mf_out_file.write("{},".format(t))
                for mf in mfs:
                    mf_out_file.write("{}".format(mf))
                    if mf != mfs[-1]:
                        mf_out_file.write(",")
                mf_out_file.write("\n")
    def get_idt_sweeps(self):
        return None

