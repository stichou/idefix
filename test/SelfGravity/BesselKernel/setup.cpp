#include "idefix.hpp"
#include "setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include <Kokkos_Complex.hpp>




using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View2D = Kokkos::View<T**, execution_space>;
template <typename T>
using View2D_host = Kokkos::View<T**, Kokkos::HostSpace>;

using axes_type = std::array<int, 2>;

/* Global class declaration */
class MyGlobalClass {
public:
  // Class constructor
  MyGlobalClass(DataBlock &data) {
  //allocate some memory for the array the class contains
    this->ibeg = data.beg[IDIR];
    this->iend = data.end[IDIR];
    this->jbeg = data.beg[JDIR];
    this->jend = data.end[JDIR];
    this->kbeg = data.beg[KDIR];
    this->kend = data.end[KDIR];

    this->n0 = kend-kbeg;
    this->n1 = jend-jbeg;
    this->n2 = iend-ibeg;
    this->n2fft = 2*n2;

    this->sigmaFFT = IdefixArray2D<real>("sigmaFFT", n1, n2fft);
    this->forceSGx = IdefixArray3D<real>("forceSGx", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    this->forceSGy = IdefixArray3D<real>("forceSGy", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    this->forceSGx_padded = IdefixArray2D<real>("forceSGx_padded", n1, n2fft);
    this->forceSGy_padded = IdefixArray2D<real>("forceSGy_padded", n1, n2fft);

    this->sigmaFFT_hat = IdefixArray2D<Kokkos::complex<real> >("sigmaFFT_hat", n1, n2fft/2+1);
    this->forceSGx_hat = IdefixArray2D<Kokkos::complex<real> >("forceSGx_hat", n1, n2fft/2+1);
    this->forceSGy_hat = IdefixArray2D<Kokkos::complex<real> >("forceSGy_hat", n1, n2fft/2+1);
    this->kernelxFFT_hat = IdefixArray2D<Kokkos::complex<real> >("kernelxFFT_hat", n1, n2fft/2+1);
    this->kernelyFFT_hat = IdefixArray2D<Kokkos::complex<real> >("kernelyFFT_hat", n1, n2fft/2+1);
  }


  /* Members of the class */
  int ibeg, jbeg, kbeg, iend, jend, kend;
  int n0, n1, n2, n2fft;

  IdefixArray2D<real> sigmaFFT;
  IdefixArray3D<real> forceSGx;
  IdefixArray3D<real> forceSGy;
  IdefixArray2D<real> forceSGx_padded;
  IdefixArray2D<real> forceSGy_padded;

  IdefixArray2D<Kokkos::complex<real> > sigmaFFT_hat;
  IdefixArray2D<Kokkos::complex<real> > kernelxFFT_hat;
  IdefixArray2D<Kokkos::complex<real> > kernelyFFT_hat;
  IdefixArray2D<Kokkos::complex<real> > forceSGx_hat;
  IdefixArray2D<Kokkos::complex<real> > forceSGy_hat;

};


/* Global class and variables */
MyGlobalClass *myGlobals;
real aspect_ratio_glob{-1};
real jump_radius_glob{-1};
real jump_width_glob{-1};
real sigma0_glob{-1};
real sigma_slope_glob{-1};
real gamma_glob{-1};
real beta_cooling_glob{-1};

// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  auto *data = hydro->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  if(dir==IDIR) {
    int ighost,ibeg,iend;
    if(side == left) {
      ighost = data->beg[IDIR];
      ibeg = 0;
      iend = data->beg[IDIR];
      idefix_for("UserDefBoundary",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          real R=x1(i);
          real Vk = 1.0/sqrt(R);

          Vc(RHO,k,j,i) = Vc(RHO,k,j,2*ighost - i +1);
          Vc(VX1,k,j,i) = - Vc(VX1,k,j,2*ighost - i +1);
          Vc(VX2,k,j,i) = Vk;
          Vc(VX3,k,j,i) = Vc(VX3,k,j,2*ighost - i +1);
        });
    }
    else if(side==right) {
      ighost = data->end[IDIR]-1;
      ibeg=data->end[IDIR];
      iend=data->np_tot[IDIR];
      idefix_for("UserDefBoundary",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          real R=x1(i);
          real Vk = 1.0/sqrt(R);

          Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
          Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
          Vc(VX2,k,j,i) = Vk;
          Vc(VX3,k,j,i) = Vc(VX3,k,j,ighost);
        });
    }
  }
}

// hydro functions to enroll
// note that everywhere we make the assumption GM = 1, so that
// Phi = - GM/R = - 1/R => Omega_K = R^(-1/2)
// cs = H * Omega_K = h * R * sqrt(GM/R^3) = h / sqrt(R)
// where R is the polar radius



void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  // locally isothermal soundspeed
  // cs = H * Omega_K
  // this is adapted from test/HD/VSI
  IdefixArray1D<real> r=data.x[IDIR];
  real aspect_ratio{aspect_ratio_glob};
  idefix_for("MySoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real R = r(i);
                real omega_k = pow(R,-1.5);
 
                cs(k,j,i) = aspect_ratio * omega_k * R; // No disc flaring
              });
}



void ComputeSgKernel(DataBlock &data) {


  int n0 = myGlobals->n0;
  int n1 = myGlobals->n1;
  int n2 = myGlobals->n2;
  int n2fft = myGlobals->n2fft;

  int jbeg = myGlobals->jbeg;

  IdefixArray1D<real> dy = data.dx[JDIR];

  real xbeg = data.xbeg[IDIR];
  real xend = data.xend[IDIR];
  IdefixArray1D<real> x=data.x[IDIR];
  IdefixArray1D<real> y=data.x[JDIR];
  IdefixArray1D<real> xFFT("x_fft", n2fft);
  IdefixArray1D<real> Hsqr_by_rpp("Hsqr_by_rpp", n2fft);
  
  IdefixArray2D<real> kernelxFFT("kernelxFFT", n1, n2fft);
  IdefixArray2D<real> kernelyFFT("kernelyFFT", n1, n2fft);
  IdefixArray2D<Kokkos::complex<real> > kernelxFFT_hat = myGlobals->kernelxFFT_hat;
  IdefixArray2D<Kokkos::complex<real> > kernelyFFT_hat = myGlobals->kernelyFFT_hat;
  IdefixArray2D<real> sigmaFFT = myGlobals->sigmaFFT;


  IdefixHostArray1D<real> xFFT_host("xFFT_host", n2fft);
  IdefixHostArray1D<real> Hsqr_by_rpp_host("Hsqr_by_rpp_host", n2fft);
  IdefixHostArray2D<real> kernelxFFT_host("kernelxFFT_host", n1, n2fft);
  IdefixHostArray2D<real> kernelyFFT_host("kernelyFFT_host", n1, n2fft);

  int ii = n2fft/2+1;
  int jj = n1/2+1;
  real alpha = xend/xbeg;
  real ratio = pow(alpha, 1.0/n2);
  real aspect_ratio{aspect_ratio_glob};
  
  IdefixHostArray1D<real> y_host("y_host", data.np_tot[JDIR]); // Include ghost cells
  auto y_mirror = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(y_mirror, y);
  Kokkos::deep_copy(y_host, y_mirror);

  IdefixHostArray1D<real> dy_host("dy_host", data.np_tot[JDIR]); // Include ghost cell
  auto dy_mirror = Kokkos::create_mirror_view(dy);
  Kokkos::deep_copy(dy_mirror, dy);
  Kokkos::deep_copy(dy_host, dy_mirror);

  int jc = data.np_tot[JDIR]/2+1;
  real y_c = y_host(jc);
  real dxFFT = pow(ratio, 0.5) - pow(ratio,-0.5); 
  real L_sg;

  execution_space exec;

  for(int i = 0; i < n2fft ; i++) {
    xFFT_host(i) = (i-ii)*log(ratio);
    Hsqr_by_rpp_host(i) = pow(aspect_ratio, 2.0)*cosh(xFFT_host(i));    
  }


  for(int j = 0; j < n1 ; j++) {
    for(int i = 0; i < n2fft ; i++) {
 
      if (i==ii and j==jj){ /* A fluid element does not feel it's own gravity */
        kernelxFFT_host(j,i) = 0;
        kernelyFFT_host(j,i) = 0;
      } else {
        real s_sqr = 2 * (  cosh(xFFT_host(i))  -  cos(y_host(jbeg+j) - y_c)  );
        real d_sqr = s_sqr / Hsqr_by_rpp_host(i);
        real ds = dxFFT * dy_host(jbeg+j); 
        real X_aux = d_sqr/8.0;
  
        if (X_aux < 60) {
            L_sg = std::pow(M_PI, 0.5)
                 * X_aux
                 * std::exp(X_aux)
                 * ( std::cyl_bessel_kl(1., X_aux)
                 - std::cyl_bessel_kl(0., X_aux) );
        } else { /* Taylor expansion at inifinity in order to avoid exp overflow  */
            L_sg = std::pow(M_PI, 0.5)
                 * X_aux
                 * 0.5 * std::pow(M_PI/2., 0.5)
                 * ( std::pow(X_aux, -1.5)
                   - 3./8.*std::pow(X_aux, -2.5)
                   + 45./128.*std::pow(X_aux, -3.5) );
        }
  
        kernelxFFT_host(j,i) = L_sg/M_PI/d_sqr * pow(exp(-xFFT_host(i))/Hsqr_by_rpp_host(i),1.5) * (exp(xFFT_host(i))-cos(y_host(jbeg+j)-y_c)) * ds;
        kernelyFFT_host(j,i) = L_sg/M_PI/d_sqr * pow(exp(-xFFT_host(i))/Hsqr_by_rpp_host(i),1.5) * sin(y_host(jbeg+j)-y_c) * ds;

        /* Debug */
        if (std::isnan(kernelxFFT_host(j, i))) { printf("NaN detected at (i=%d, j=%d)\n", i, j);}
      }
    }
  }

  /* Copy host arrays into device arrays */
  auto kernelxFFT_mirror = Kokkos::create_mirror_view(kernelxFFT);
  Kokkos::deep_copy(kernelxFFT_mirror, kernelxFFT_host);
  Kokkos::deep_copy(kernelxFFT, kernelxFFT_mirror);

  auto kernelyFFT_mirror = Kokkos::create_mirror_view(kernelyFFT);
  Kokkos::deep_copy(kernelyFFT_mirror, kernelyFFT_host);
  Kokkos::deep_copy(kernelyFFT, kernelyFFT_mirror);


  /* 2D Forward Transform */
  KokkosFFT::rfft2(exec, kernelxFFT, kernelxFFT_hat);
  KokkosFFT::rfft2(exec, kernelyFFT, kernelyFFT_hat);

  ///* ********************************************************** */
  ///* Sanity check : FFT^-1(FFT(kernel)) = kernel                */
  ///* ********************************************************** */
  //IdefixArray3D<real> kernelx_out("kernelx_out", n0, n1, n2);
  //IdefixArray3D<real> kernely_out("kernely_out", n0, n1, n2);

  //KokkosFFT::irfft2(exec, kernelxFFT_hat, kernelxFFT);
  //KokkosFFT::irfft2(exec, kernelyFFT_hat, kernelyFFT);

  //idefix_for("FFT_loop", 0, n0, 0, n1, 0, n2,
  //            KOKKOS_LAMBDA (int k, int j, int i) {
  //              kernelx_out(k,j,i) = kernelxFFT(j,i) ;
  //              kernely_out(k,j,i) = kernelyFFT(j,i) ;
  //            });

  //idfx::DumpArray("kernelx.npy", kernelx_out); 
  //idfx::DumpArray("kernely.npy", kernely_out); 
  ///* ********************************************************** */

}



void ComputeSgForces(DataBlock &data) {
  IdefixArray4D<real> Vc=data.hydro->Vc;

  int n0 = myGlobals->n0;
  int n1 = myGlobals->n1;
  int n2 = myGlobals->n2;
  int n2fft = myGlobals->n2fft;

  int ibeg = myGlobals->ibeg;
  int jbeg = myGlobals->jbeg;
  int iend = myGlobals->iend;

  IdefixArray2D<Kokkos::complex<real> > kernelxFFT_hat = myGlobals->kernelxFFT_hat;
  IdefixArray2D<Kokkos::complex<real> > kernelyFFT_hat = myGlobals->kernelyFFT_hat;
  IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
  IdefixArray3D<real> forceSGy = myGlobals->forceSGy;
  IdefixArray2D<real> sigmaFFT = myGlobals->sigmaFFT;
  IdefixArray2D<Kokkos::complex<real> > sigmaFFT_hat = myGlobals->sigmaFFT_hat;
  IdefixArray2D<Kokkos::complex<real> > forceSGx_hat = myGlobals->forceSGx_hat;
  IdefixArray2D<Kokkos::complex<real> > forceSGy_hat = myGlobals->forceSGy_hat;
  IdefixArray2D<real> forceSGx_padded = myGlobals->forceSGx_padded;
  IdefixArray2D<real> forceSGy_padded = myGlobals->forceSGy_padded;

  real G = 1.0; /* TODO: What is its value ? */

  execution_space exec;

  /* Compute kernel only once */
  static auto init = [&] {
    idfx::pushRegion("Self-gravity spectral: kernel (init)");   // Profiling and debugging
    ComputeSgKernel(data); 
    idfx::popRegion();
    return 0; // Non-void return required for static variable
  }();

  /* Create plans only once */

  static KokkosFFT::Plan forward_plan(
        execution_space(),              
        sigmaFFT,                       
        sigmaFFT_hat,                   
        KokkosFFT::Direction::forward,  
        /*axes=*/axes_type({0, 1}) 
    );

  static KokkosFFT::Plan backward_plan(
        execution_space(),              
        forceSGx_hat,                   
        forceSGx_padded,                
        KokkosFFT::Direction::backward, 
        /*axes=*/axes_type({0, 1}) 
    );


  /* Fill sigma */
  idefix_for("FFT_loop", 0, n0, 0, n1, 0, n2fft,
              KOKKOS_LAMBDA (int k, int j, int i) {
                if (i<iend) {
                  sigmaFFT(j,i) = Vc(RHO,k,j,i) ;
                } else {
                  sigmaFFT(j,i) = 0.0 ;
                }
              });


  /* 2D Forward Transform */
  KokkosFFT::execute(forward_plan, sigmaFFT, sigmaFFT_hat); 

  /* Product in Fourier space */
  idefix_for("FFT_loop", 0, n1, 0, n2fft / 2 + 1,
              KOKKOS_LAMBDA(int j, int i) {
        forceSGx_hat(j, i) = -G * kernelxFFT_hat(j, i) * sigmaFFT_hat(j, i);
        forceSGy_hat(j, i) = -G * kernelyFFT_hat(j, i) * sigmaFFT_hat(j, i);
    }
  ); 

  /* 2D Backward Transform */
  KokkosFFT::execute(backward_plan, forceSGx_hat, forceSGx_padded);
  KokkosFFT::execute(backward_plan, forceSGy_hat, forceSGy_padded);


  ///* ********************************************************** */
  ///* Sanity check : FFT^-1(FFT(sigma)) = sigma                  */
  ///* ********************************************************** */
  //KokkosFFT::irfft2(exec, sigmaFFT_hat, sigmaFFT);
  //idfx::DumpArray("sigmaFFT.npy", sigmaFFT); 
  ///* ********************************************************** */


  /* Copy padded arrays into standard Idefix arrays */
  idefix_for("FFT_loop", 0, n0, 0, n1, 0, n2,
              KOKKOS_LAMBDA (int k, int j, int i) {
	        int i_offset = i+n2;
                forceSGx(k,jbeg+j,i+ibeg) = forceSGx_padded(j,i_offset) ;
                forceSGy(k,jbeg+j,i+ibeg) = forceSGy_padded(j,i_offset) ;
              });

}






/* Fargo velocity=Keplerian rotation */
void FargoVelocity(DataBlock &data, IdefixArray2D<real> &Vphi) {
  IdefixArray1D<real> x1 = data.x[IDIR];

  idefix_for("FargoVphi",0,data.np_tot[KDIR], 0, data.np_tot[IDIR],
      KOKKOS_LAMBDA (int k, int i) {
      Vphi(k,i) = 1.0/sqrt(x1(i));
  });
}



void ComputeUserVars(DataBlock &data, UserDefVariablesContainer &variables) {
  // Mirror data on host
  DataBlockHost d(data);
 
  // Sync it
  d.SyncFromDevice();

  IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
  IdefixArray3D<real> forceSGy = myGlobals->forceSGy;

  IdefixHostArray3D<real> forceSGx_host = variables["forceSGx"];
  IdefixHostArray3D<real> forceSGy_host = variables["forceSGy"];

  Kokkos::deep_copy(forceSGx_host, forceSGx);
  Kokkos::deep_copy(forceSGy_host, forceSGy);

  /* Toomre parameter */
  IdefixHostArray3D<real> toomre_host = variables["toomre"];
  real aspect_ratio{aspect_ratio_glob};

#ifndef ISOTHERMAL
  real gamma{gamma_glob};
#endif
  
  for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
    for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
      for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
        real r = d.x[IDIR](i);
        real omega_k = pow(r,-1.5);

#ifndef ISOTHERMAL
        real cs=sqrt(gamma* d.Vc(PRS,k,j,i)/d.Vc(RHO,k,j,i));
#else
        real cs = omega_k * aspect_ratio * r;
#endif

        /* Self-gravity contribution */
        toomre_host(k,j,i) = cs*omega_k/M_PI/d.Vc(RHO,k,j,i);
      }
    }
  }


}



void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;  // Main cell-centered primitive variables index
  IdefixArray4D<real> Uc = hydro->Uc;  // Main cell-centered conservative variables
#ifndef ISOTHERMAL
  IdefixArray1D<real> x1 = data->x[IDIR];
  real gamma{gamma_glob};
  real sigma0{sigma0_glob};
  real sigma_slope{sigma_slope_glob};
  real aspect_ratio{aspect_ratio_glob};
#endif

  real dt = dtin;
  real xbeg = data->xbeg[IDIR];
  real xend = data->xend[IDIR];
  real omega_in = sqrt(1.0/pow(xbeg,3.0));
  real t_in = 2 * M_PI / omega_in; 
  real t_0 = 110 * t_in;
  real t_ramp = 5*t_in;

  real beta_cooling_target{beta_cooling_glob};
  real beta_cooling = 30.0 + 0.5*(1+tanh(t-5960.0))*(beta_cooling_target-30.0);

  
  //printf("beta_cooling=%f\n", beta_cooling);

  idfx::pushRegion("Self-gravity spectral: force");
  ComputeSgForces(*data); 
  idfx::popRegion();

  IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
  IdefixArray3D<real> forceSGy = myGlobals->forceSGy;


  idefix_for("MySgSourceTerm",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                Uc(MX1,k,j,i) += Vc(RHO,k,j,i)*forceSGx(k,j,i)*dt;
                Uc(MX2,k,j,i) += Vc(RHO,k,j,i)*forceSGy(k,j,i)*dt;
#ifndef ISOTHERMAL
                real R = x1(i);
                real omega_k = pow(R,-1.5);
                real cs0 = aspect_ratio * R * omega_k;
                real RHO0 = sigma0 * pow(R, sigma_slope);
                real PRS0 = RHO0*cs0*cs0/gamma;
                //real PRS0 = 0.0;

                Uc(ENG, k,j,i) += Vc(RHO,k,j,i)*(forceSGx(k,j,i)*Vc(VX1,k,j,i) + forceSGy(k,j,i)*Vc(VX2,k,j,i)) * dt;
                Uc(ENG, k,j,i) += -dt*(Vc(PRS,k,j,i)-0.001*PRS0)/(gamma-1.0)*omega_k/beta_cooling;
#endif
  
  });

}



Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    data.hydro->EnrollUserDefBoundary(&UserdefBoundary);  // BC for gas
#ifdef ISOTHERMAL
    data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
#endif

    aspect_ratio_glob = input.Get<real>("Setup","aspect_ratio",0);
    jump_radius_glob = input.Get<real>("Setup", "jump_radius",0);
    jump_width_glob = input.Get<real>("Setup", "jump_width",0);
    sigma0_glob = input.Get<real>("Setup", "sigma0", 0);
    sigma_slope_glob = input.Get<real>("Setup", "sigma_slope", 0);
    beta_cooling_glob = input.Get<real>("Setup", "beta_cooling", 0);
#ifndef ISOTHERMAL
    gamma_glob = data.hydro->eos->GetGamma();
#endif

    /* Initialise Global variables */
    myGlobals = new MyGlobalClass(data);



    if(data.haveFargo)
      data.fargo->EnrollVelocity(&FargoVelocity);
    
    data.hydro->EnrollUserSourceTerm(&MySourceTerm);
    output.EnrollUserDefVariables(&ComputeUserVars);
    
}



void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    // arbitrary values for now, I'll set those as parameters later
    real jump_width{jump_width_glob};
    real jump_radius{jump_radius_glob};
    real aspect_ratio{aspect_ratio_glob};
    real sigma0{sigma0_glob};
    real sigma_slope{sigma_slope_glob};
#ifndef ISOTHERMAL
    real gamma{gamma_glob};
#endif

    /* Initialise density */
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
      for(int j = 0; j < d.np_tot[JDIR] ; j++) {
        for(int i = 0; i < d.np_tot[IDIR] ; i++) {
          real r = d.x[IDIR](i);
          real X = (r-jump_radius)/jump_width;

          d.Vc(RHO,k,j,i) = sigma0 * pow(r, sigma_slope) ; 
          //d.Vc(RHO,k,j,i) = sigma0 * pow(r, sigma_slope) * 0.5 * (1.00001 + tanh(X)) ; 

        }
      }
    }

    // Send it all, if needed
    d.SyncToDevice();

    /* Compute self-gravity */
    idfx::pushRegion("Self-gravity spectral: force (init)");   // Profiling and debugging
    ComputeSgForces(data); 
    idfx::popRegion();

    IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
    IdefixHostArray3D<real> forceSGx_host("forceSGx_host (init)", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    auto forceSGx_mirror = Kokkos::create_mirror_view(forceSGx);
    Kokkos::deep_copy(forceSGx_mirror, forceSGx);
    Kokkos::deep_copy(forceSGx_host, forceSGx_mirror);

    /* Update velocity profile to be at centrifugal equilibrium */
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
      for(int j = 0; j < d.np_tot[JDIR] ; j++) {
        for(int i = 0; i < d.np_tot[IDIR] ; i++) {

          real r = d.x[IDIR](i);
          real omega_k = pow(r,-1.5);
          real vk_sqr = pow(omega_k*r,2.0);
          real vphi_sqr;
          real p = sigma_slope;
          real diffrSigma_by_Sigma;
          real X = (r-jump_radius)/jump_width;
          
          /* simple density power law */
          vphi_sqr = vk_sqr 
                   + pow(aspect_ratio, 2.0) * (sigma_slope-1) / r 
                   - r*forceSGx_host(k,j,i);

          /* Tapered density in the inner region (J. A. Rendon Restrepo) */
          //diffrSigma_by_Sigma = p/r + 1./jump_width/pow(cosh(X),2.0)/(1.00001 + tanh(X));
          //vphi_sqr = vk_sqr \
          //           + pow(aspect_ratio, 2.0)*(diffrSigma_by_Sigma-1/r) \
          //           - r*forceSGx_host(k,j,i);

          /* Debug */
          if (vphi_sqr<0.0 && i>=data.beg[IDIR] && i<=data.end[IDIR]){ 
            printf("Negative vphi_sqr detected at (i=%d, j=%d)\n", i, j);
          }

          /* Velocity field */
          d.Vc(VX1,k,j,i) = 0.0;
          d.Vc(VX2,k,j,i) = sqrt(vphi_sqr);

#ifndef ISOTHERMAL
          real cs = aspect_ratio * r * omega_k;
          d.Vc(PRS,k,j,i) = d.Vc(RHO,k,j,i)*cs*cs/gamma;
#endif
          // add some random noise to the radial velocity component break the
          // axial symmetry and let the instability grow
          d.Vc(VX1,k,j,i) = d.Vc(VX2,k,j,i) * aspect_ratio * 1e-1*(0.5-idfx::randm());
        }
      }
    }

    // Send it all, if needed
    d.SyncToDevice();

}


// Analyse data to produce an output
void MakeAnalysis(DataBlock & data) {

}



// Do a specifically designed user step in the middle of the integration
void ComputeUserStep(DataBlock &data, real t, real dt) {

}



/* Setup destructor */
Setup::~Setup() {
    delete myGlobals;
}




