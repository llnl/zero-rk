

#include "radau_cpp.h"

#include <assert.h>
#include <fenv.h>
#include <cmath>
#include <stdio.h>

namespace radau_cpp {

radau::radau(int _n, N_Vector _y0)
    : n(_n),
      nfcn(0),
      njac(0),
      nstep(0),
      naccept(0),
      nreject(0),
      ndec(0),
      nsol(0),
      ns(0),
      nsmin(3),
      nsmax(7),
      nsus(nsmin),
      nmax(100000),
      nit(7),
      startn(false),
      pred(true),
      uround(1.0e-16),  // TODO:
      h(0.0),
      hacc(0.0),
      hmax(0.0),
      thet(0.001),
      quot1(1.0),
      quot2(1.2),
      facl(5.0),
      facr(1.0 / 8.0),
      safe(0.9),
      vitu(0.002),
      vitd(0.8),
      hhou(1.2),
      hhod(0.8),
      // atol(1.0e-10),
      // rtol(1.0e-6),
      caljac(false),
      reject(false),
      first(false),
      last(false),
      err(0.0),
      erracc(0.0),
      x(0.0),
      deriv_fcn(NULL),
      jac_fcn(NULL),
      jac_decomp_fcn(NULL),
      jac_solve_fcn(NULL),
      jac_complex_decomp_fcn(NULL),
      jac_complex_solve_fcn(NULL),
      output_fcn(NULL),
      output_fcn_data(NULL),
      user_data(NULL) {
  rtol = N_VClone(_y0);
  atol = N_VClone(_y0);
}

radau::~radau() {
  deriv_fcn = NULL;
  jac_fcn = NULL;
  jac_decomp_fcn = NULL;
  jac_solve_fcn = NULL;
  jac_complex_decomp_fcn = NULL;
  jac_complex_solve_fcn = NULL;
  output_fcn = NULL;
  output_fcn_data = NULL;
  user_data = NULL;
  N_VDestroy(rtol);
  N_VDestroy(atol);
}

void radau::set_nmax(int _nmax) {
  assert(_nmax >= 0);
  nmax = _nmax;
}

void radau::set_uround(double _uround) {
  assert(_uround > 0. && _uround < 1.);
  uround = _uround;
}

void radau::set_hinit(double _hinit) {
  assert(_hinit != 0.);
  h = _hinit;
}

void radau::set_hmax(double _hmax) {
  assert(_hmax != 0.);
  hmax = _hmax;
}

void radau::set_thet(double _thet) {
  assert(_thet > 0. && _thet < 1.);
  thet = _thet;
}

void radau::set_step_size_params(double _facl, double _facr) {
  facl = _facl;
  facr = _facr;
}

void radau::set_quot_factors(double _quot1, double _quot2) {
  assert(_quot1 <= 1.0);
  assert(_quot2 >= 1.0);
  quot1 = _quot1;
  quot2 = _quot2;
}

void radau::set_safe(double _safe) { safe = _safe; }

void radau::set_order_params(double _vitu, double _vitd, double _hhou,
                             double _hhod) {
  vitu = _vitu;
  vitd = _vitd;
  hhou = _hhou;
  hhod = _hhod;
}

void radau::set_nsmin(int _nsmin) {
  nsmin = std::max(1, _nsmin);
  if (_nsmin >= 2) nsmin = 3;
  if (_nsmin >= 4) nsmin = 5;
  if (_nsmin >= 6) nsmin = 7;
  nsus = nsmin;
}

void radau::set_nsmax(int _nsmax) {
  nsmax = std::min(7, _nsmax);
  if (_nsmax <= 6) nsmax = 5;
  if (_nsmax <= 4) nsmax = 3;
  if (_nsmax <= 2) nsmax = 1;
}

void radau::set_nsus(int _nsus) {
  nsus = _nsus;
  if (nsus == 1 || nsus == 3 || nsus == 5 || nsus || 7) {
    // pass
  } else {
    assert(false);
  }
}

void radau::set_nit(int _nit) {
  assert(_nit > 0 && _nit <= 50);
  nit = _nit;
}

void radau::set_startn(bool _startn) { startn = _startn; }

void radau::set_pred(bool _pred) { pred = _pred; }

void radau::set_tolerances(N_Vector _rtol, N_Vector _atol) {
  N_VScale(1.0, _rtol, rtol);
  N_VScale(1.0, _atol, atol);
  check_tolerances();
}

void radau::set_tolerances(double _rtol, double _atol) {
  N_VConst(_rtol, rtol);
  N_VConst(_atol, atol);
  check_tolerances();
}

void radau::check_tolerances() {
  double min_atol = N_VMin(atol);
  double min_rtol = N_VMin(rtol);
  assert(min_atol > 0.0);
  assert(min_rtol > 10.0 * uround);
}

void radau::set_deriv_fcn(radau_deriv_fcn _deriv_fcn) {
  deriv_fcn = _deriv_fcn;
}

void radau::set_jac_fcn(radau_jac_fcn _jac_fcn) { jac_fcn = _jac_fcn; }

void radau::set_jac_decomp_fcn(radau_jac_decomp_fcn _jac_decomp_fcn) {
  jac_decomp_fcn = _jac_decomp_fcn;
}

void radau::set_jac_solve_fcn(radau_jac_solve_fcn _jac_solve_fcn) {
  jac_solve_fcn = _jac_solve_fcn;
}

void radau::set_jac_complex_decomp_fcn(
    radau_jac_complex_decomp_fcn _jac_complex_decomp_fcn) {
  jac_complex_decomp_fcn = _jac_complex_decomp_fcn;
}

void radau::set_jac_complex_solve_fcn(
    radau_jac_complex_solve_fcn _jac_complex_solve_fcn) {
  jac_complex_solve_fcn = _jac_complex_solve_fcn;
}

void radau::set_output_fcn(radau_output_fcn _output_fcn,
                           void* _output_fcn_data) {
  output_fcn = _output_fcn;
  output_fcn_data = _output_fcn_data;
}

void radau::set_user_data(void* _user_data) { user_data = _user_data; }

void radau::get_integrator_stats(int* _nfcn, int* _njac, int* _nstep,
                                 int* _naccept, int* _nreject, int* _ndec,
                                 int* _nsol) {
  *_nfcn = nfcn;
  *_njac = njac;
  *_nstep = nstep;
  *_naccept = naccept;
  *_nreject = nreject;
  *_ndec = ndec;
  *_nsol = nsol;
}

int radau::solve(double* x_in, double xend, N_Vector y) {
  // reset counters
  nfcn = 0;
  njac = 0;
  nstep = 0;
  naccept = 0;
  nreject = 0;
  ndec = 0;
  nsol = 0;

  // ns = nsmax;
  ns = nsus;

  assert(deriv_fcn != NULL);
  assert(jac_fcn != NULL);
  assert(jac_decomp_fcn != NULL);
  assert(jac_solve_fcn != NULL);

  create_work_vectors(y);

  x = *x_in;

  caljac = false;
  reject = false;
  first = true;
  last = false;

  // -------- CHECK THE INDEX OF THE PROBLEM -----
  bool variab = nsmin < nsmax;
  int idid = -1;
  int nit1 = nit;
  double expo = 1.0 / (ns + 1.0);
  double sq6 = sqrt(6.0);
  double c31 = (4.0 - sq6) / 10.0;
  double c32 = (4.0 + sq6) / 10.0;
  double c31m1 = c31 - 1.0;
  double c32m1 = c32 - 1.0;
  double c31mc2 = c31 - c32;
  bool unexp = false;
  bool unexn = false;
  bool change = false;
  int ikeep = 0;
  int ichan = 0;
  double theta = 0.00;
  double thetat = 0.00;
  coercv(ns);
  double posneg = xend - x > 0 ? 1.0 : -1.0;
  if (hmax == 0.0) hmax = xend - x;
  double hmaxn = std::min(std::abs(hmax), std::abs(xend - x));
  if (std::abs(h) <= 10.0 * uround) {
    h = 1.0e-6;
  }
  h = posneg * std::min(std::abs(h), hmaxn);
  double hold = h;
  double hnew = h;
  double fac;
  double quot;

  int newt = 0;

  if ((x + h * 1.0001 - xend) * posneg >= 0.0) {
    h = xend - x;
    last = true;
  }

  double hopt = h;
  double faccon = 1.0;
  int nsing = 0;
  double xold = x;
  double xph = x + h;
  double dynold;
  double thqold;

  if (output_fcn != NULL) {
    // TODO (zz isn't right here...)
    output_fcn(nstep, x, h, y, zz[0], output_fcn_data);
  }

  double expmns = (ns + 1.0) / (2.0 * ns);
  double rtol1, atol1;
  double quott = N_VMin(atol) / N_VMin(rtol);
  rtol1 = 0.1 * std::pow(N_VMin(rtol), expmns);
  atol1 = rtol1 * quott;
  N_VAbs(y, scal);
  N_VScale(rtol1, scal, scal);
  N_VAddConst(scal, atol1, scal);
  N_VInv(scal, scal);  // N.B. we work with inverse scal.

  double hhfac = h;
  deriv_fcn(x, y, y0, user_data);
  nfcn = nfcn + 1;

// --- basic integration step
g10:
  // --- compute jacobian matrix with user supplied function
  jac_fcn(x, y, y0, user_data);
  ++njac;
  caljac = true;
g20:
  // --- change the order here if necessary
  if (variab) {
    int nsnew = ns;
    ichan = ichan + 1;
    double hquot = h / hold;
    thetat = std::min(10.0, std::max(theta, thetat * 0.5));
    if (newt > 1 && thetat <= vitu && hquot < hhou && hquot > hhod) {
      nsnew = std::min(nsmax, ns + 2);
    }
    if (thetat >= vitd || unexp) nsnew = std::max(nsmin, ns - 2);
    if (ichan >= 1 && unexn) nsnew = std::max(nsmin, ns - 2);
    if (ichan <= 10) nsnew = std::min(ns, nsnew);
    change = ns != nsnew;
    unexn = false;
    unexp = false;
    if (change) {
      ns = nsnew;
      ichan = 1;
      coercv(ns);
      expo = 1.0 / (ns + 1.0);
      expmns = (ns + 1.0) / (2.0 * ns);
      quott = N_VMin(atol) / N_VMin(rtol);
      rtol1 = 0.1 * std::pow(N_VMin(rtol), expmns);
      atol1 = rtol1 * quott;
      N_VAbs(y, scal);
      N_VScale(rtol1, scal, scal);
      N_VAddConst(scal, atol1, scal);
      N_VInv(scal, scal);  // N.B. we work with inverse scal.
    }
  }

  // --- compute the matrices e1 and e2 and their decompositions
  // double fac1=u1/h;
  double fac1 = h / u1;
  int ier = jac_decomp_fcn(fac1, user_data);
  if (ier != 0) goto g78;

  for (int k = 0; k < (ns - 1) / 2; ++k) {
    alphn[k] = alph[k] / h;
    betan[k] = beta[k] / h;
    int ier = jac_complex_decomp_fcn(k, alphn[k], betan[k], user_data);
    if (ier != 0) goto g78;
  }
  ++ndec;

g30:
  if (variab && ikeep == 1) {
    ichan = ichan + 1;
    ikeep = 0;
    if (ichan >= 10 && ns < nsmax) goto g20;
  }
  ++nstep;

  // Check for failure
  if (nstep > nmax) {
    printf(" Exit of RadauCPP at x=%18.4f\n", x);
    printf(" MORE THAN NMAX = %d STEPS ARE NEEDED\n", nmax);
    idid = -2;
    *x_in = x;
    destroy_work_vectors();
    return idid;
  }
  if (0.1 * std::abs(h) <= std::abs(x) * uround) {
    printf(" Exit of RadauCPP at x=%18.4f\n", x);
    printf(" STEP SIZE T0O SMALL, H=%g\n", h);
    idid = -3;
    *x_in = x;
    destroy_work_vectors();
    return idid;
  }

  xph = x + h;

  // *** *** *** *** *** *** ***
  // *** *** *** *** *** *** ***
  if (ns == 3) {
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***
    if (first || startn || change) {
      for (int i = 0; i < ns; ++i) {
        N_VConst(0.0, zz[i]);
        N_VConst(0.0, ff[i]);
      }
    } else {
      double hquot = h / hold;
      double c3q = hquot;
      double c1q = c31 * c3q;
      double c2q = c32 * c3q;
      N_VLinearSum(1.0, cont[2], (c1q - c31m1), cont[3], zz[0]);
      N_VLinearSum(1.0, cont[1], (c1q - c32m1), zz[0], zz[0]);
      N_VScale(c1q, zz[0], zz[0]);

      N_VLinearSum(1.0, cont[2], (c2q - c31m1), cont[3], zz[1]);
      N_VLinearSum(1.0, cont[1], (c2q - c32m1), zz[1], zz[1]);
      N_VScale(c2q, zz[1], zz[1]);

      N_VLinearSum(1.0, cont[2], (c3q - c31m1), cont[3], zz[2]);
      N_VLinearSum(1.0, cont[1], (c3q - c32m1), zz[2], zz[2]);
      N_VScale(c3q, zz[2], zz[2]);

      N_VLinearSum(ti312, zz[1], ti313, zz[2], ff[0]);
      N_VLinearSum(ti311, zz[0], 1.0, ff[0], ff[0]);

      N_VLinearSum(ti322, zz[1], ti323, zz[2], ff[1]);
      N_VLinearSum(ti321, zz[0], 1.0, ff[1], ff[1]);

      N_VLinearSum(ti332, zz[1], ti333, zz[2], ff[2]);
      N_VLinearSum(ti331, zz[0], 1.0, ff[2], ff[2]);
    }
    // *** *** *** *** *** *** ***
    //  loop for the simplified newton iteration
    // *** *** *** *** *** *** ***
    newt = 0;
    nit = nit1;
    double expmi = 1.0 / expmns;
    double fnewt = std::max(10 * uround / rtol1,
                            std::min(0.03, std::pow(rtol1, (expmi - 1.0))));
    faccon = std::pow(std::max(faccon, uround), 0.8);
    theta = std::abs(thet);

  g40:
    if (newt >= nit) goto g78;
    // ---     compute the right-hand side
    N_VLinearSum(1.0, y, 1.0, zz[0], cont[0]);
    deriv_fcn(x + c31 * h, cont[0], zz[0], user_data);
    N_VLinearSum(1.0, y, 1.0, zz[1], cont[0]);
    deriv_fcn(x + c32 * h, cont[0], zz[1], user_data);
    N_VLinearSum(1.0, y, 1.0, zz[2], cont[0]);
    deriv_fcn(xph, cont[0], zz[2], user_data);
    nfcn = nfcn + 3;

    // ---     solve the linear systems
    N_VLinearSum(ti311, zz[0], ti312, zz[1], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti313, zz[2], tmp[0]);
    N_VLinearSum(ti321, zz[0], ti322, zz[1], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti323, zz[2], tmp[1]);
    N_VLinearSum(ti331, zz[0], ti332, zz[1], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti333, zz[2], tmp[2]);
    N_VScale(1.0, tmp[0], zz[0]);
    N_VScale(1.0, tmp[1], zz[1]);
    N_VScale(1.0, tmp[2], zz[2]);
    slvrad(fac1, alphn[0], betan[0], zz[0], zz[1], zz[2], ff[0], ff[1], ff[2],
           cont[0]);
    nsol = nsol + 1;
    newt = newt + 1;
    double dyno1 = N_VWL2Norm(zz[0], scal);
    double dyno2 = N_VWL2Norm(zz[1], scal);
    double dyno3 = N_VWL2Norm(zz[2], scal);
    double dyno =
        sqrt((dyno1 * dyno1 + dyno2 * dyno2 + dyno3 * dyno3) / (n * ns));
    // ---     bad convergence or number of iterations too large
    if (newt > 1 && newt < nit) {
      double thq = dyno / dynold;
      if (newt == 2) {
        theta = thq;
      } else {
        theta = sqrt(thq * thqold);
      }
      thqold = thq;
      if (theta < 0.99) {
        faccon = theta / (1.0 - theta);
        double dyth = faccon * dyno * std::pow(theta, (nit - 1 - newt)) / fnewt;
        if (dyth >= 1.0) {
          double qnewt = std::max(1.0e-4, std::min(20.0, dyth));
          hhfac = 0.8 * std::pow(qnewt, (-1.0 / (4.0 + nit - 1 - newt)));
          h = hhfac * h;
          reject = true;
          last = false;
          if (hhfac <= 0.5) unexn = true;
          if (caljac) goto g20;
          goto g10;
        }
      } else {
        goto g78;
      }
    }
    dynold = std::max(dyno, uround);
    N_VLinearSum(1.0, ff[0], 1.0, zz[0], ff[0]);
    N_VLinearSum(1.0, ff[1], 1.0, zz[1], ff[1]);
    N_VLinearSum(1.0, ff[2], 1.0, zz[2], ff[2]);
    N_VLinearSum(t311, ff[0], t312, ff[1], zz[0]);
    N_VLinearSum(1.0, zz[0], t313, ff[2], zz[0]);
    N_VLinearSum(t321, ff[0], t322, ff[1], zz[1]);
    N_VLinearSum(1.0, zz[1], t323, ff[2], zz[1]);
    N_VLinearSum(t331, ff[0], 1.0, ff[1], zz[2]);
    // nsol 17, ff differs
    if (faccon * dyno > fnewt) goto g40;
    // --- error estimation
    err = estrad(y);
    if (std::isnan(err)) goto g78;  // test
    // --- compute finite differences for dense output
    if (err < 1.0) {
      N_VLinearSum(1.0, y, 1.0, zz[2], y);
      N_VLinearSum(1.0, zz[1], -1.0, zz[2], cont[1]);
      N_VScale(1.0 / c32m1, cont[1], cont[1]);
      N_VLinearSum(1.0 / c31mc2, zz[0], -1.0 / c31mc2, zz[1], tmp[0]);  // ak
      N_VScale(1.0 / c31, zz[0], tmp[1]);  // acont3
      N_VLinearSum(1.0 / c32, tmp[0], -1.0 / c32, tmp[1], tmp[1]);
      N_VLinearSum(1.0 / c31m1, tmp[0], -1.0 / c31m1, cont[1], cont[2]);
      N_VLinearSum(1.0, cont[2], -1.0, tmp[1], cont[3]);
    }
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***

  } else if (ns == 5) {
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***
    if (first || startn || change) {
      for (int i = 0; i < ns; ++i) {
        N_VConst(0.0, zz[i]);
        N_VConst(0.0, ff[i]);
      }
    } else {
      double hquot = h / hold;
      for (int k = 0; k < ns; ++k) {
        double ccq = c[k + 1] * hquot;
        N_VScale(1.0, cont[ns], zz[k]);
        for (int l = ns - 2; l >= 0; --l) {
          N_VLinearSum(1.0, cont[l + 1], ccq - c[ns - 1 - l] + 1.0, zz[k],
                       zz[k]);
        }
        N_VScale(ccq, zz[k], zz[k]);
      }
      N_VLinearSum(ti511, zz[0], ti512, zz[1], ff[0]);
      N_VLinearSum(1.0, ff[0], ti513, zz[2], ff[0]);
      N_VLinearSum(1.0, ff[0], ti514, zz[3], ff[0]);
      N_VLinearSum(1.0, ff[0], ti515, zz[4], ff[0]);

      N_VLinearSum(ti521, zz[0], ti522, zz[1], ff[1]);
      N_VLinearSum(1.0, ff[1], ti523, zz[2], ff[1]);
      N_VLinearSum(1.0, ff[1], ti524, zz[3], ff[1]);
      N_VLinearSum(1.0, ff[1], ti525, zz[4], ff[1]);

      N_VLinearSum(ti531, zz[0], ti532, zz[1], ff[2]);
      N_VLinearSum(1.0, ff[2], ti533, zz[2], ff[2]);
      N_VLinearSum(1.0, ff[2], ti534, zz[3], ff[2]);
      N_VLinearSum(1.0, ff[2], ti535, zz[4], ff[2]);

      N_VLinearSum(ti541, zz[0], ti542, zz[1], ff[3]);
      N_VLinearSum(1.0, ff[3], ti543, zz[2], ff[3]);
      N_VLinearSum(1.0, ff[3], ti544, zz[3], ff[3]);
      N_VLinearSum(1.0, ff[3], ti545, zz[4], ff[3]);

      N_VLinearSum(ti551, zz[0], ti552, zz[1], ff[4]);
      N_VLinearSum(1.0, ff[4], ti553, zz[2], ff[4]);
      N_VLinearSum(1.0, ff[4], ti554, zz[3], ff[4]);
      N_VLinearSum(1.0, ff[4], ti555, zz[4], ff[4]);
    }
    // *** *** *** *** *** *** ***
    //  LOOP FOR THE SIMPLIFIED NEWTON ITERATION
    // *** *** *** *** *** *** ***
    newt = 0;
    nit = nit1 + 5;
    double expmi = 1.0 / expmns;
    double fnewt = std::max(10 * uround / rtol1,
                            std::min(0.03, std::pow(rtol1, (expmi - 1.0))));
    faccon = std::pow(std::max(faccon, uround), 0.8);
    theta = std::abs(thet);

  g140:
    if (newt >= nit) goto g78;
    // ---     compute the right-hand side
    for (int k = 0; k < ns; ++k) {
      N_VLinearSum(1.0, y, 1.0, zz[k], cont[0]);
      deriv_fcn(x + c[k + 1] * h, cont[0], zz[k], user_data);
    }
    nfcn = nfcn + 5;
    // ---     solve the linear systems
    N_VLinearSum(ti511, zz[0], ti512, zz[1], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti513, zz[2], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti514, zz[3], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti515, zz[4], tmp[0]);

    N_VLinearSum(ti521, zz[0], ti522, zz[1], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti523, zz[2], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti524, zz[3], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti525, zz[4], tmp[1]);

    N_VLinearSum(ti531, zz[0], ti532, zz[1], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti533, zz[2], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti534, zz[3], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti535, zz[4], tmp[2]);

    N_VLinearSum(ti541, zz[0], ti542, zz[1], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti543, zz[2], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti544, zz[3], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti545, zz[4], tmp[3]);

    N_VLinearSum(ti551, zz[0], ti552, zz[1], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti553, zz[2], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti554, zz[3], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti555, zz[4], tmp[4]);

    N_VScale(1.0, tmp[0], zz[0]);
    N_VScale(1.0, tmp[1], zz[1]);
    N_VScale(1.0, tmp[2], zz[2]);
    N_VScale(1.0, tmp[3], zz[3]);
    N_VScale(1.0, tmp[4], zz[4]);

    slvrar(fac1, zz[0], ff[0]);
    for (int k = 0; k < 2; ++k) {
      slvrai(k, alphn[k], betan[k], zz[2 * k + 1], zz[2 * k + 2], ff[2 * k + 1],
             ff[2 * k + 2], cont[0]);
    }
    ++nsol;
    ++newt;
    double dyno1 = N_VWL2Norm(zz[0], scal);
    double dyno2 = N_VWL2Norm(zz[1], scal);
    double dyno3 = N_VWL2Norm(zz[2], scal);
    double dyno4 = N_VWL2Norm(zz[3], scal);
    double dyno5 = N_VWL2Norm(zz[4], scal);
    double dyno = sqrt((dyno1 * dyno1 + dyno2 * dyno2 + dyno3 * dyno3 +
                        dyno4 * dyno4 + dyno5 * dyno5) /
                       (n * ns));
    // TODO: Consider numerical issues with above...
    //---     BAD CONVERGENCE OR NUMBER OF ITERATIONS TO LARGE
    if (newt > 1 && newt < nit) {
      double thq = dyno / dynold;
      if (newt == 2) {
        theta = thq;
      } else {
        theta = sqrt(thq * thqold);
      }
      thqold = thq;
      if (theta < 0.99) {
        faccon = theta / (1.0 - theta);
        double dyth = faccon * dyno * std::pow(theta, (nit - 1 - newt)) / fnewt;
        if (dyth >= 1.0) {
          double qnewt = std::max(1.0e-4, std::min(20.0, dyth));
          hhfac = 0.8 * std::pow(qnewt, (-1.0 / (4.0 + nit - 1 - newt)));
          h = hhfac * h;
          reject = true;
          last = false;
          if (hhfac <= 0.5) unexn = true;
          if (caljac) goto g20;
          goto g10;
        }
      } else {
        goto g78;
      }
    }
    dynold = std::max(dyno, uround);
    N_VLinearSum(1.0, ff[0], 1.0, zz[0], ff[0]);
    N_VLinearSum(1.0, ff[1], 1.0, zz[1], ff[1]);
    N_VLinearSum(1.0, ff[2], 1.0, zz[2], ff[2]);
    N_VLinearSum(1.0, ff[3], 1.0, zz[3], ff[3]);
    N_VLinearSum(1.0, ff[4], 1.0, zz[4], ff[4]);

    N_VLinearSum(t511, ff[0], t512, ff[1], zz[0]);
    N_VLinearSum(1.0, zz[0], t513, ff[2], zz[0]);
    N_VLinearSum(1.0, zz[0], t514, ff[3], zz[0]);
    N_VLinearSum(1.0, zz[0], t515, ff[4], zz[0]);

    N_VLinearSum(t521, ff[0], t522, ff[1], zz[1]);
    N_VLinearSum(1.0, zz[1], t523, ff[2], zz[1]);
    N_VLinearSum(1.0, zz[1], t524, ff[3], zz[1]);
    N_VLinearSum(1.0, zz[1], t525, ff[4], zz[1]);

    N_VLinearSum(t531, ff[0], t532, ff[1], zz[2]);
    N_VLinearSum(1.0, zz[2], t533, ff[2], zz[2]);
    N_VLinearSum(1.0, zz[2], t534, ff[3], zz[2]);
    N_VLinearSum(1.0, zz[2], t535, ff[4], zz[2]);

    N_VLinearSum(t541, ff[0], t542, ff[1], zz[3]);
    N_VLinearSum(1.0, zz[3], t543, ff[2], zz[3]);
    N_VLinearSum(1.0, zz[3], t544, ff[3], zz[3]);
    N_VLinearSum(1.0, zz[3], t545, ff[4], zz[3]);

    N_VLinearSum(t551, ff[0], 1.0, ff[1], zz[4]);
    N_VLinearSum(1.0, zz[4], 1.0, ff[3], zz[4]);

    if (faccon * dyno > fnewt) goto g140;
    // --- error estimation
    err = estrav(y);
    if (std::isnan(err)) goto g78;  // test
    //       --- compute finite differences for dense output
    //       TODO: PROBLEM HERE?
    if (err < 1.0) {
      N_VLinearSum(1.0, y, 1.0, zz[4], y);
      N_VScale(1.0 / c[1], zz[0], cont[5]);
      for (int k = 0; k < ns - 1; ++k) {
        double fact = 1.0 / (c[ns - 1 - k] - c[ns - k]);
        N_VLinearSum(fact, zz[ns - k - 2], -fact, zz[ns - k - 1], cont[k + 1]);
      }
      for (int j = 1; j < ns; ++j) {
        for (int k = ns - 1; k >= j; --k) {
          double fact = 1.0 / (c[ns - k - 1] - c[ns - k + j]);
          N_VLinearSum(fact, cont[k + 1], -fact, cont[k], cont[k + 1]);
        }
      }
    }
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***

  } else if (ns == 7) {
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***

    if (first || startn || change) {
      for (int i = 0; i < ns; ++i) {
        N_VConst(0.0, zz[i]);
        N_VConst(0.0, ff[i]);
      }
    } else {
      double hquot = h / hold;
      for (int k = 0; k < ns; ++k) {
        double ccq = c[k + 1] * hquot;
        N_VScale(1.0, cont[ns], zz[k]);
        for (int l = ns - 2; l >= 0; --l) {
          N_VLinearSum(1.0, cont[l + 1], ccq - c[ns - 1 - l] + 1.0, zz[k],
                       zz[k]);
        }
        N_VScale(ccq, zz[k], zz[k]);
      }
      N_VLinearSum(ti711, zz[0], ti712, zz[1], ff[0]);
      N_VLinearSum(1.0, ff[0], ti713, zz[2], ff[0]);
      N_VLinearSum(1.0, ff[0], ti714, zz[3], ff[0]);
      N_VLinearSum(1.0, ff[0], ti715, zz[4], ff[0]);
      N_VLinearSum(1.0, ff[0], ti716, zz[5], ff[0]);
      N_VLinearSum(1.0, ff[0], ti717, zz[6], ff[0]);

      N_VLinearSum(ti721, zz[0], ti722, zz[1], ff[1]);
      N_VLinearSum(1.0, ff[1], ti723, zz[2], ff[1]);
      N_VLinearSum(1.0, ff[1], ti724, zz[3], ff[1]);
      N_VLinearSum(1.0, ff[1], ti725, zz[4], ff[1]);
      N_VLinearSum(1.0, ff[1], ti726, zz[5], ff[1]);
      N_VLinearSum(1.0, ff[1], ti727, zz[6], ff[1]);

      N_VLinearSum(ti731, zz[0], ti732, zz[1], ff[2]);
      N_VLinearSum(1.0, ff[2], ti733, zz[2], ff[2]);
      N_VLinearSum(1.0, ff[2], ti734, zz[3], ff[2]);
      N_VLinearSum(1.0, ff[2], ti735, zz[4], ff[2]);
      N_VLinearSum(1.0, ff[2], ti736, zz[5], ff[2]);
      N_VLinearSum(1.0, ff[2], ti737, zz[6], ff[2]);

      N_VLinearSum(ti741, zz[0], ti742, zz[1], ff[3]);
      N_VLinearSum(1.0, ff[3], ti743, zz[2], ff[3]);
      N_VLinearSum(1.0, ff[3], ti744, zz[3], ff[3]);
      N_VLinearSum(1.0, ff[3], ti745, zz[4], ff[3]);
      N_VLinearSum(1.0, ff[3], ti746, zz[5], ff[3]);
      N_VLinearSum(1.0, ff[3], ti747, zz[6], ff[3]);

      N_VLinearSum(ti751, zz[0], ti752, zz[1], ff[4]);
      N_VLinearSum(1.0, ff[4], ti753, zz[2], ff[4]);
      N_VLinearSum(1.0, ff[4], ti754, zz[3], ff[4]);
      N_VLinearSum(1.0, ff[4], ti755, zz[4], ff[4]);
      N_VLinearSum(1.0, ff[4], ti756, zz[5], ff[4]);
      N_VLinearSum(1.0, ff[4], ti757, zz[6], ff[4]);

      N_VLinearSum(ti761, zz[0], ti762, zz[1], ff[5]);
      N_VLinearSum(1.0, ff[5], ti763, zz[2], ff[5]);
      N_VLinearSum(1.0, ff[5], ti764, zz[3], ff[5]);
      N_VLinearSum(1.0, ff[5], ti765, zz[4], ff[5]);
      N_VLinearSum(1.0, ff[5], ti766, zz[5], ff[5]);
      N_VLinearSum(1.0, ff[5], ti767, zz[6], ff[5]);

      N_VLinearSum(ti771, zz[0], ti772, zz[1], ff[6]);
      N_VLinearSum(1.0, ff[6], ti773, zz[2], ff[6]);
      N_VLinearSum(1.0, ff[6], ti774, zz[3], ff[6]);
      N_VLinearSum(1.0, ff[6], ti775, zz[4], ff[6]);
      N_VLinearSum(1.0, ff[6], ti776, zz[5], ff[6]);
      N_VLinearSum(1.0, ff[6], ti777, zz[6], ff[6]);
    }
    // *** *** *** *** *** *** ***
    //  loop for the simplified newton iteration
    // *** *** *** *** *** *** ***
    newt = 0;
    nit = nit1 + 10;
    double expmi = 1.0 / expmns;
    double fnewt = std::max(10 * uround / rtol1,
                            std::min(0.03, std::pow(rtol1, (expmi - 1.0))));
    faccon = std::pow(std::max(faccon, uround), 0.8);
    theta = std::abs(thet);
  g240:
    if (newt >= nit) goto g78;
    // ---     compute the right-hand side
    for (int k = 0; k < ns; ++k) {
      N_VLinearSum(1.0, y, 1.0, zz[k], cont[0]);
      deriv_fcn(x + c[k + 1] * h, cont[0], zz[k], user_data);
    }
    nfcn = nfcn + ns;
    // ---     solve the linear systems
    N_VLinearSum(ti711, zz[0], ti712, zz[1], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti713, zz[2], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti714, zz[3], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti715, zz[4], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti716, zz[5], tmp[0]);
    N_VLinearSum(1.0, tmp[0], ti717, zz[6], tmp[0]);

    N_VLinearSum(ti721, zz[0], ti722, zz[1], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti723, zz[2], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti724, zz[3], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti725, zz[4], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti726, zz[5], tmp[1]);
    N_VLinearSum(1.0, tmp[1], ti727, zz[6], tmp[1]);

    N_VLinearSum(ti731, zz[0], ti732, zz[1], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti733, zz[2], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti734, zz[3], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti735, zz[4], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti736, zz[5], tmp[2]);
    N_VLinearSum(1.0, tmp[2], ti737, zz[6], tmp[2]);

    N_VLinearSum(ti741, zz[0], ti742, zz[1], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti743, zz[2], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti744, zz[3], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti745, zz[4], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti746, zz[5], tmp[3]);
    N_VLinearSum(1.0, tmp[3], ti747, zz[6], tmp[3]);

    N_VLinearSum(ti751, zz[0], ti752, zz[1], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti753, zz[2], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti754, zz[3], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti755, zz[4], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti756, zz[5], tmp[4]);
    N_VLinearSum(1.0, tmp[4], ti757, zz[6], tmp[4]);

    N_VLinearSum(ti761, zz[0], ti762, zz[1], tmp[5]);
    N_VLinearSum(1.0, tmp[5], ti763, zz[2], tmp[5]);
    N_VLinearSum(1.0, tmp[5], ti764, zz[3], tmp[5]);
    N_VLinearSum(1.0, tmp[5], ti765, zz[4], tmp[5]);
    N_VLinearSum(1.0, tmp[5], ti766, zz[5], tmp[5]);
    N_VLinearSum(1.0, tmp[5], ti767, zz[6], tmp[5]);

    N_VLinearSum(ti771, zz[0], ti772, zz[1], tmp[6]);
    N_VLinearSum(1.0, tmp[6], ti773, zz[2], tmp[6]);
    N_VLinearSum(1.0, tmp[6], ti774, zz[3], tmp[6]);
    N_VLinearSum(1.0, tmp[6], ti775, zz[4], tmp[6]);
    N_VLinearSum(1.0, tmp[6], ti776, zz[5], tmp[6]);
    N_VLinearSum(1.0, tmp[6], ti777, zz[6], tmp[6]);

    N_VScale(1.0, tmp[0], zz[0]);
    N_VScale(1.0, tmp[1], zz[1]);
    N_VScale(1.0, tmp[2], zz[2]);
    N_VScale(1.0, tmp[3], zz[3]);
    N_VScale(1.0, tmp[4], zz[4]);
    N_VScale(1.0, tmp[5], zz[5]);
    N_VScale(1.0, tmp[6], zz[6]);

    slvrar(fac1, zz[0], ff[0]);
    for (int k = 0; k < 3; ++k) {
      slvrai(k, alphn[k], betan[k], zz[2 * k + 1], zz[2 * k + 2], ff[2 * k + 1],
             ff[2 * k + 2], cont[0]);
    }
    ++nsol;
    ++newt;
    double dyno1 = N_VWL2Norm(zz[0], scal);
    double dyno2 = N_VWL2Norm(zz[1], scal);
    double dyno3 = N_VWL2Norm(zz[2], scal);
    double dyno4 = N_VWL2Norm(zz[3], scal);
    double dyno5 = N_VWL2Norm(zz[4], scal);
    double dyno6 = N_VWL2Norm(zz[5], scal);
    double dyno7 = N_VWL2Norm(zz[6], scal);
    double dyno =
        sqrt((dyno1 * dyno1 + dyno2 * dyno2 + dyno3 * dyno3 + dyno4 * dyno4 +
              dyno5 * dyno5 + dyno6 * dyno6 + dyno7 * dyno7) /
             (n * ns));
    // TODO: Consider numerical issues with above...
    // ---     bad convergence or number of iterations to large
    if (newt > 1 && newt < nit) {
      double thq = dyno / dynold;
      if (newt == 2) {
        theta = thq;
      } else {
        theta = sqrt(thq * thqold);
      }
      thqold = thq;
      if (theta < 0.99) {
        faccon = theta / (1.0 - theta);
        double dyth = faccon * dyno * std::pow(theta, (nit - 1 - newt)) / fnewt;
        if (dyth >= 1.0) {
          double qnewt = std::max(1.0e-4, std::min(20.0, dyth));
          hhfac = 0.8 * std::pow(qnewt, (-1.0 / (4.0 + nit - 1 - newt)));
          h = hhfac * h;
          reject = true;
          last = false;
          if (hhfac <= 0.5) unexn = true;
          if (caljac) goto g20;
          goto g10;
        }
      } else {
        goto g78;
      }
    }
    dynold = std::max(dyno, uround);
    N_VLinearSum(1.0, ff[0], 1.0, zz[0], ff[0]);
    N_VLinearSum(1.0, ff[1], 1.0, zz[1], ff[1]);
    N_VLinearSum(1.0, ff[2], 1.0, zz[2], ff[2]);
    N_VLinearSum(1.0, ff[3], 1.0, zz[3], ff[3]);
    N_VLinearSum(1.0, ff[4], 1.0, zz[4], ff[4]);
    N_VLinearSum(1.0, ff[5], 1.0, zz[5], ff[5]);
    N_VLinearSum(1.0, ff[6], 1.0, zz[6], ff[6]);

    N_VLinearSum(t711, ff[0], t712, ff[1], zz[0]);
    N_VLinearSum(1.0, zz[0], t713, ff[2], zz[0]);
    N_VLinearSum(1.0, zz[0], t714, ff[3], zz[0]);
    N_VLinearSum(1.0, zz[0], t715, ff[4], zz[0]);
    N_VLinearSum(1.0, zz[0], t716, ff[5], zz[0]);
    N_VLinearSum(1.0, zz[0], t717, ff[6], zz[0]);

    N_VLinearSum(t721, ff[0], t722, ff[1], zz[1]);
    N_VLinearSum(1.0, zz[1], t723, ff[2], zz[1]);
    N_VLinearSum(1.0, zz[1], t724, ff[3], zz[1]);
    N_VLinearSum(1.0, zz[1], t725, ff[4], zz[1]);
    N_VLinearSum(1.0, zz[1], t726, ff[5], zz[1]);
    N_VLinearSum(1.0, zz[1], t727, ff[6], zz[1]);

    N_VLinearSum(t731, ff[0], t732, ff[1], zz[2]);
    N_VLinearSum(1.0, zz[2], t733, ff[2], zz[2]);
    N_VLinearSum(1.0, zz[2], t734, ff[3], zz[2]);
    N_VLinearSum(1.0, zz[2], t735, ff[4], zz[2]);
    N_VLinearSum(1.0, zz[2], t736, ff[5], zz[2]);
    N_VLinearSum(1.0, zz[2], t737, ff[6], zz[2]);

    N_VLinearSum(t741, ff[0], t742, ff[1], zz[3]);
    N_VLinearSum(1.0, zz[3], t743, ff[2], zz[3]);
    N_VLinearSum(1.0, zz[3], t744, ff[3], zz[3]);
    N_VLinearSum(1.0, zz[3], t745, ff[4], zz[3]);
    N_VLinearSum(1.0, zz[3], t746, ff[5], zz[3]);
    N_VLinearSum(1.0, zz[3], t747, ff[6], zz[3]);

    N_VLinearSum(t751, ff[0], t752, ff[1], zz[4]);
    N_VLinearSum(1.0, zz[4], t753, ff[2], zz[4]);
    N_VLinearSum(1.0, zz[4], t754, ff[3], zz[4]);
    N_VLinearSum(1.0, zz[4], t755, ff[4], zz[4]);
    N_VLinearSum(1.0, zz[4], t756, ff[5], zz[4]);
    N_VLinearSum(1.0, zz[4], t757, ff[6], zz[4]);

    N_VLinearSum(t761, ff[0], t762, ff[1], zz[5]);
    N_VLinearSum(1.0, zz[5], t763, ff[2], zz[5]);
    N_VLinearSum(1.0, zz[5], t764, ff[3], zz[5]);
    N_VLinearSum(1.0, zz[5], t765, ff[4], zz[5]);
    N_VLinearSum(1.0, zz[5], t766, ff[5], zz[5]);
    N_VLinearSum(1.0, zz[5], t767, ff[6], zz[5]);

    N_VLinearSum(t771, ff[0], 1.0, ff[1], zz[6]);
    N_VLinearSum(1.0, zz[6], 1.0, ff[3], zz[6]);
    N_VLinearSum(1.0, zz[6], 1.0, ff[5], zz[6]);

    if (faccon * dyno > fnewt) goto g240;
    // --- error estimation
    err = estrav(y);
    if (std::isnan(err)) goto g78;  // test
    // --- compute finite differences for dense output
    if (err < 1.0) {
      N_VLinearSum(1.0, y, 1.0, zz[6], y);
      N_VScale(1.0 / c[1], zz[0], cont[7]);
      for (int k = 0; k < ns - 1; ++k) {
        double fact = 1.0 / (c[ns - 1 - k] - c[ns - k]);
        N_VLinearSum(fact, zz[ns - k - 2], -fact, zz[ns - k - 1], cont[k + 1]);
      }
      for (int j = 1; j < ns; ++j) {
        for (int k = ns - 1; k >= j; --k) {
          double fact = 1.0 / (c[ns - k - 1] - c[ns - k + j]);
          N_VLinearSum(fact, cont[k + 1], -fact, cont[k], cont[k + 1]);
        }
      }
    }

    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***

  } else if (ns == 1) {
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***
    if (first || startn || change) {
      for (int i = 0; i < ns; ++i) {
        N_VConst(0.0, zz[i]);
        N_VConst(0.0, ff[i]);
      }
    } else {
      double hquot = h / hold;
      N_VScale(hquot, cont[1], zz[0]);
      N_VScale(1.0, zz[0], ff[0]);
    }
    // *** *** *** *** *** *** ***
    //  loop for the simplified newton iteration
    // *** *** *** *** *** *** ***
    newt = 0;
    nit = nit1 - 3;
    double expmi = 1.0 / expmns;
    double fnewt = std::max(10 * uround / rtol1, 0.03);
    faccon = std::pow(std::max(faccon, uround), 0.8);
    theta = std::abs(thet);
  g440:
    if (newt >= nit) goto g78;
    // ---     compute the right-hand side
    N_VLinearSum(1.0, y, 1.0, zz[0], cont[0]);
    deriv_fcn(xph, cont[0], zz[0], user_data);
    ++nfcn;
    // ---     solve the linear systems
    slvrar(fac1, zz[0], ff[0]);
    ++nsol;
    ++newt;
    double dyno = N_VWrmsNorm(zz[0], scal);
    // ---     BAD CONVERGENCE OR NUMBER OF ITERATIONS TO LARGE
    if (newt > 1 && newt < nit) {
      double thq = dyno / dynold;
      if (newt == 2) {
        theta = thq;
      } else {
        theta = sqrt(thq * thqold);
      }
      thqold = thq;
      if (theta < 0.99) {
        faccon = theta / (1.0 - theta);
        double dyth = faccon * dyno * std::pow(theta, (nit - 1 - newt)) / fnewt;
        if (dyth >= 1.0) {
          double qnewt = std::max(1.0e-4, std::min(20.0, dyth));
          hhfac = 0.8 * std::pow(qnewt, (-1.0 / (4.0 + nit - 1 - newt)));
          h = hhfac * h;
          reject = true;
          last = false;
          if (hhfac <= 0.5) unexn = true;
          if (caljac) goto g20;
          goto g10;
        }
      } else {
        goto g78;
      }
    }
    dynold = std::max(dyno, uround);
    N_VLinearSum(1.0, ff[0], 1.0, zz[0], ff[0]);
    N_VScale(1.0, ff[0], zz[0]);
    if (faccon * dyno > fnewt) goto g440;
    // --- error estimation
    err = estrav(y);
    if (std::isnan(err)) goto g78;  // test
    // --- compute finite differences for dense output
    if (err < 1.0) {
      N_VLinearSum(1.0, y, 1.0, zz[0], y);
      N_VScale(1.0, zz[0], cont[1]);
    }
    // *** *** *** *** *** *** ***
    // *** *** *** *** *** *** ***
  }  // if ns ==
  // *** *** *** *** *** *** ***
  // *** *** *** *** *** *** ***
  // --- computation of hnew
  // --- we require .2<=hnew/h<=8.
  fac = std::min(safe, (1 + 2 * nit) * safe / (newt + 2 * nit));
  quot = std::max(facr, std::min(facl, std::pow(err, expo) / fac));
  hnew = h / quot;
  // *** *** *** *** *** *** ***
  //  is the error small enough ?
  // *** *** *** *** *** *** ***
  if (err < 1.0) {
    // --- step is accepted
    first = false;
    naccept = naccept + 1;
    if (pred && !change) {
      //    --- predictive controller of gustafsson
      if (naccept > 1) {
        double facgus = (hacc / h) * std::pow(err * err / erracc, expo) / safe;
        facgus = std::max(facr, std::min(facl, facgus));
        quot = std::max(quot, facgus);
        hnew = h / quot;
      }
      hacc = h;
      erracc = std::max(1.0e-2, err);
    }
    xold = x;
    hold = h;
    x = xph;
    //    --- update scaling
    N_VAbs(y, scal);
    N_VScale(rtol1, scal, scal);
    N_VAddConst(scal, atol1, scal);
    N_VInv(scal, scal);  // N.B. we work with inverse scal.
    if (output_fcn != NULL) {
      N_VScale(1.0, y, cont[0]);
      output_fcn(naccept, x, h, y, y0, output_fcn_data);
    }
    caljac = false;
    if (last) {
      h = hopt;
      idid = 0;  // 1;
      *x_in = x;
      destroy_work_vectors();
      return idid;
    }
    deriv_fcn(x, y, y0, user_data);
    ++nfcn;
    hnew = posneg * std::min(std::abs(hnew), hmaxn);
    hopt = hnew;
    hopt = std::min(h, hnew);
    if (reject) hnew = posneg * std::min(std::abs(hnew), std::abs(h));
    reject = false;
    if ((x + hnew / quot1 - xend) * posneg >= 0.0) {
      h = xend - x;
      last = true;
    } else {
      double qt = hnew / h;
      hhfac = h;
      if (theta <= thet && qt >= quot1 && qt <= quot2) {
        ikeep = 1;
        goto g30;
      }
      h = hnew;
    }
    hhfac = h;
    if (theta <= thet) goto g20;
    goto g10;
  } else {
    // --- STEP IS REJECTED
    reject = true;
    last = false;
    if (first) {
      h = h * 0.1;
      hhfac = 0.1;
    } else {
      hhfac = hnew / h;
      h = hnew;
    }
    if (naccept >= 1) nreject = nreject + 1;
    if (caljac) goto g20;
    goto g10;
  }
// --- unexpected step-rejection
g78:
  unexp = true;
  if (ier != 0) {
    nsing = nsing + 1;
    if (nsing >= 5) {
      printf(" Exit of RadauCPP at x=%18.4f\n", x);
      printf(" MATRIX IS REPEATEDLY SINGULAR, IER=%d\n", ier);
      idid = -4;
      *x_in = x;
      destroy_work_vectors();
      return idid;
    }
  }
  h = h * 0.5;
  hhfac = 0.5;
  reject = true;
  last = false;
  if (caljac) goto g20;
  goto g10;
g179:
  printf(" Exit of RadauCPP at x=%18.4f\n", x);
  idid = 2;
  *x_in = x;
  destroy_work_vectors();
  return idid;
}

void radau::coercv(int ns) {
  c[0] = 0.0;
  c[ns] = 1.0;
  if (ns == 1) {
    c[0] = 1.0;
    u1 = 1.0;
    dd[0] = -1.0;
  } else if (ns == 3) {
    double sq6 = sqrt(6.0);
    c[1] = (4.0 - sq6) / 10.0;
    c[2] = (4.0 + sq6) / 10.0;
    dd[0] = -(13.0 + 7.0 * sq6) / 3.0;
    dd[1] = (-13.0 + 7.0 * sq6) / 3.0;
    dd[2] = -1.0 / 3.0;
    double st9 = std::pow(9.0, (1.0 / 3.0));
    u1 = (6.0 + st9 * (st9 - 1)) / 30.0;
    double alp = (12.0 - st9 * (st9 - 1)) / 60.0;
    double bet = st9 * (st9 + 1) * sqrt(3.0) / 60.0;
    double cno = alp * alp + bet * bet;
    u1 = 1.0 / u1;
    alph[0] = alp / cno;
    beta[0] = bet / cno;
  } else if (ns == 5) {
    c[1] = 0.5710419611451768219312e-01;
    c[2] = 0.2768430136381238276800e+00;
    c[3] = 0.5835904323689168200567e+00;
    c[4] = 0.8602401356562194478479e+00;
    dd[0] = -0.2778093394406463730479e+02;
    dd[1] = 0.3641478498049213152712e+01;
    dd[2] = -0.1252547721169118720491e+01;
    dd[3] = 0.5920031671845428725662e+00;
    dd[4] = -0.2000000000000000000000e+00;
    u1 = 0.6286704751729276645173e+01;
    alph[0] = 0.3655694325463572258243e+01;
    beta[0] = 0.6543736899360077294021e+01;
    alph[1] = 0.5700953298671789419170e+01;
    beta[1] = 0.3210265600308549888425e+01;
  } else if (ns == 7) {
    c[1] = 0.2931642715978489197205e-01;
    c[2] = 0.1480785996684842918500e+00;
    c[3] = 0.3369846902811542990971e+00;
    c[4] = 0.5586715187715501320814e+00;
    c[5] = 0.7692338620300545009169e+00;
    c[6] = 0.9269456713197411148519e+00;
    dd[0] = -0.5437443689412861451458e+02;
    dd[1] = 0.7000024004259186512041e+01;
    dd[2] = -0.2355661091987557192256e+01;
    dd[3] = 0.1132289066106134386384e+01;
    dd[4] = -0.6468913267673587118673e+00;
    dd[5] = 0.3875333853753523774248e+00;
    dd[6] = -0.1428571428571428571429e+00;
    u1 = 0.8936832788405216337302e+01;
    alph[0] = 0.4378693561506806002523e+01;
    beta[0] = 0.1016969328379501162732e+02;
    alph[1] = 0.7141055219187640105775e+01;
    beta[1] = 0.6623045922639275970621e+01;
    alph[2] = 0.8511834825102945723051e+01;
    beta[2] = 0.3281013624325058830036e+01;
  }
}

int radau::slvrai(int k, double alpha, double beta, N_Vector z2, N_Vector z3,
                  N_Vector f2, N_Vector f3, N_Vector cc) {
  N_VScale(-1, f2, tmp[0]);  // S2->tmp[0]
  N_VScale(-1, f3, tmp[1]);  // S3->tmp[1]
  N_VLinearSum(1.0, z2, alpha, tmp[0], z2);
  N_VLinearSum(1.0, z2, -beta, tmp[1], z2);
  N_VLinearSum(1.0, z3, alpha, tmp[1], cc);
  N_VLinearSum(1.0, cc, beta, tmp[0], cc);
  int ier = jac_complex_solve_fcn(k, z2, cc, user_data);
  // CALL ZGETRS ('No transpose',N,1,E2R,LDE1,IP2,Z2,N,IER)
  N_VScale(1.0, cc, z3);
  return ier;
}

int radau::slvrad(double fac1, double alpha, double beta, N_Vector z1,
                  N_Vector z2, N_Vector z3, N_Vector f1, N_Vector f2,
                  N_Vector f3, N_Vector cc) {
  N_VLinearSum(h / u1, z1, -1.0, f1, z1);
  int ier1 = jac_solve_fcn(x, y, y0, z1, tmp1, user_data);
  N_VScale(1.0, tmp1, z1);

  N_VScale(-1, f2, tmp[0]);  // S2->tmp[0]
  N_VScale(-1, f3, tmp[1]);  // S3->tmp[1]
  N_VLinearSum(1.0, z2, alpha, tmp[0], z2);
  N_VLinearSum(1.0, z2, -beta, tmp[1], z2);
  N_VLinearSum(1.0, z3, alpha, tmp[1], cc);
  N_VLinearSum(1.0, cc, beta, tmp[0], cc);
  int ier2 = jac_complex_solve_fcn(0, z2, cc, user_data);
  N_VScale(1.0, cc, z3);
  if (ier1 > 0 || ier2 > 0) {
    return std::max(ier1, ier2);
  } else {
    return std::min(ier1, ier2);
  }
}

int radau::slvrar(double fac1, N_Vector z1, N_Vector f1) {
  N_VLinearSum(h / u1, z1, -1.0, f1, z1);
  // int ier = jac_solve_fcn(z1,user_data);
  int ier = jac_solve_fcn(x, y, y0, z1, tmp1, user_data);
  N_VScale(1.0, tmp1, z1);

  return ier;
}

double radau::estrav(N_Vector y) {
  double err = 10.0;
  N_VConst(0.0, ff[1]);
  for (int k = 0; k < ns; ++k) {
    N_VLinearSum(1.0, ff[1], dd[k], zz[k], ff[1]);
  }
  N_VScale(1.0 / h, ff[1], ff[1]);
  N_VLinearSum(1.0, ff[1], 1.0, y0, cont[0]);
  N_VScale(h / u1, cont[0], cont[0]);
  // jac_solve_fcn(cont[0],user_data);
  int ier1 = jac_solve_fcn(x, y, y0, cont[0], tmp1, user_data);
  N_VScale(1.0, tmp1, cont[0]);

  err = N_VWrmsNorm(cont[0], scal);
  err = std::max(err, 1.0e-10);
  if (err < 1.0) return err;

  if (first || reject) {
    N_VLinearSum(1.0, y, 1.0, cont[0], cont[0]);
    deriv_fcn(x, cont[0], ff[0], user_data);
    ++nfcn;
    N_VLinearSum(1.0, ff[0], 1.0, ff[1], cont[0]);
    N_VScale(h / u1, cont[0], cont[0]);
    // jac_solve_fcn(cont[0],user_data);
    jac_solve_fcn(x, y, y0, cont[0], tmp1, user_data);
    N_VScale(1.0, tmp1, cont[0]);

    err = N_VWrmsNorm(cont[0], scal);
    err = std::max(err, 1.0e-10);
  }
  return err;
}

// TODO: Should be equivalent to estrav
double radau::estrad(N_Vector y) {
  double err = 10.0;
  double hee1 = dd[0] / h;
  double hee2 = dd[1] / h;
  double hee3 = dd[2] / h;
  N_VLinearSum(hee1, zz[0], hee2, zz[1], ff[1]);
  N_VLinearSum(1.0, ff[1], hee3, zz[2], ff[1]);
  N_VLinearSum(1.0, ff[1], 1.0, y0, cont[0]);
  N_VScale(h / u1, cont[0], cont[0]);
  // jac_solve_fcn(cont[0],user_data);
  jac_solve_fcn(x, y, y0, cont[0], tmp1, user_data);
  N_VScale(1.0, tmp1, cont[0]);

  err = N_VWrmsNorm(cont[0], scal);
  err = std::max(err, 1.0e-10);
  if (err < 1.0) return err;

  if (first || reject) {
    N_VLinearSum(1.0, y, 1.0, cont[0], cont[0]);
    deriv_fcn(x, cont[0], ff[0], user_data);
    ++nfcn;
    N_VLinearSum(1.0, ff[0], 1.0, ff[1], cont[0]);
    N_VScale(h / u1, cont[0], cont[0]);
    // jac_solve_fcn(cont[0],user_data);
    jac_solve_fcn(x, y, y0, cont[0], tmp1, user_data);
    N_VScale(1.0, tmp1, cont[0]);

    err = N_VWrmsNorm(cont[0], scal);
    err = std::max(err, 1.0e-10);
  }
  return err;
}

void radau::contra(const double xout, N_Vector yout) {
  // ----------------------------------------------------------
  //     THIS FUNCTION CAN BE USED FOR CONINUOUS OUTPUT. IT PROVIDES AN
  //     APPROXIMATION OF THE SOLUTION AT X.
  //     IT GIVES THE VALUE OF THE COLLOCATION POLYNOMIAL, DEFINED FOR
  //     THE LAST SUCCESSFULLY COMPUTED STEP (BY RADAU).
  // ----------------------------------------------------------
  double s = (xout - x) / h + 1.0;  // TODO: Not sure if using correct h
  N_VScale(1.0, cont[ns], yout);
  for (int k = ns - 1; k >= 0; --k) {
    N_VLinearSum(1.0, cont[k], s - c[ns - k], yout, yout);
  }
  return;
}

void radau::create_work_vectors(N_Vector y) {
  y0 = N_VClone(y);
  N_VConst(0.0, y0);
  scal = N_VClone(y);
  N_VConst(0.0, scal);
  tmp1 = N_VClone(y);
  N_VConst(0.0, tmp1);
  tmp2 = N_VClone(y);
  N_VConst(0.0, tmp2);
  tmp3 = N_VClone(y);
  N_VConst(0.0, tmp3);
  ff.resize(nsmax);
  zz.resize(nsmax);
  cont.resize(nsmax + 1);
  tmp.resize(nsmax);
  for (int i = 0; i < nsmax; ++i) {
    ff[i] = N_VClone(y);
    N_VConst(0.0, ff[i]);
    zz[i] = N_VClone(y);
    N_VConst(0.0, zz[i]);
    tmp[i] = N_VClone(y);
    N_VConst(0.0, tmp[i]);
  }
  for (int i = 0; i <= nsmax; ++i) {
    cont[i] = N_VClone(y);
    N_VConst(0.0, cont[i]);
  }
}

void radau::destroy_work_vectors() {
  N_VDestroy(y0);
  N_VDestroy(scal);
  N_VDestroy(tmp1);
  N_VDestroy(tmp2);
  N_VDestroy(tmp3);
  for (int i = 0; i < nsmax; ++i) {
    N_VDestroy(ff[i]);
    N_VDestroy(zz[i]);
    N_VDestroy(tmp[i]);
  }
  for (int i = 0; i <= nsmax; ++i) {
    N_VDestroy(cont[i]);
  }
}

}  // end namespace radau_cpp
