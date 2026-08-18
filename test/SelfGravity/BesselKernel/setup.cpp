#include "idefix.hpp"
#include "setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include <optional>



using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View2D = Kokkos::View<T**, execution_space>;
template <typename T>
using View2D_host = Kokkos::View<T**, Kokkos::HostSpace>;

using axes_type = std::array<int, 2>;

/* Global class declaration */
class MyGlobalClass {
private:
  int n0, n1, n2, n2fft;
  
public:
  // Class constructor
  MyGlobalClass(DataBlock &data) {
  //allocate some memory for the array the class contains
    this->n0 = data.np_tot[KDIR];
    this->n1 = data.np_tot[JDIR];
    this->n2 = data.np_tot[IDIR];
    this->n2fft = 2*data.np_tot[IDIR];

    this->kernelxFFT_hat = View2D<Kokkos::complex<real> >("kernelxFFT_hat", n1, n2fft/2+1);
    this->kernelyFFT_hat = View2D<Kokkos::complex<real> >("kernelyFFT_hat", n1, n2fft/2+1);
    this->forceSGx = IdefixArray3D<real>("forceSGx", n0, n1, n2);
    this->forceSGy = IdefixArray3D<real>("forceSGy", n0, n1, n2);

    this->sigmaFFT = View2D<real>("sigmaFFT", n1, n2fft);
    this->sigmaFFT_hat = View2D<Kokkos::complex<real> >("sigmaFFT_hat", n1, n2fft/2+1);
    this->forceSGx_hat = View2D<Kokkos::complex<real> >("forceSGx_hat", n1, n2fft/2+1);
    this->forceSGy_hat = View2D<Kokkos::complex<real> >("forceSGy_hat", n1, n2fft/2+1);
    this->forceSGx_padded = View2D<real>("forceSGx_padded", n1, n2fft);
    this->forceSGy_padded = View2D<real>("forceSGy_padded", n1, n2fft);

  }


  /* members of the class */
  View2D<Kokkos::complex<real> > kernelxFFT_hat;
  View2D<Kokkos::complex<real> > kernelyFFT_hat;
  IdefixArray3D<real> forceSGx;
  IdefixArray3D<real> forceSGy;


  View2D<real> sigmaFFT;
  View2D<Kokkos::complex<real> > sigmaFFT_hat;
  View2D<Kokkos::complex<real> > forceSGx_hat;
  View2D<Kokkos::complex<real> > forceSGy_hat;
  View2D<real> forceSGx_padded;
  View2D<real> forceSGy_padded;

};


/* Global class and variables */
MyGlobalClass *myGlobals;
real aspect_ratio_glob{-1};
real jump_radius_glob{-1};
real jump_width_glob{-1};



// hydro functions to enroll
// note that everywhere we make the assumption GM = 1, so that
// Phi = - GM/R = - 1/R => Omega_K = R^(-1/2)
// cs = H * Omega_K = h * R * sqrt(GM/R^3) = h / sqrt(R)
// where R is the polar radius



void LISOTHSoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  // locally isothermal soundspeed
  // cs = H * Omega_K
  // this is adapted from test/HD/VSI
  IdefixArray1D<real> r=data.x[IDIR];
  real aspect_ratio{aspect_ratio_glob};
  idefix_for("LISOTHSoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real R = r(i);
                cs(k,j,i) = aspect_ratio/sqrt(R);
              });
}



void ComputeSgKernel(DataBlock &data) {

  idfx::pushRegion("SG kernel");   // Profiling and debugging

  int n0=data.np_tot[KDIR];
  int n1=data.np_tot[JDIR];
  int n2=data.np_tot[IDIR];
  int n2fft=2*data.np_tot[IDIR];

  IdefixArray1D<real> dy = data.dx[JDIR];

  real xbeg = data.xbeg[IDIR];
  real xend = data.xend[IDIR];
  IdefixArray1D<real> x=data.x[IDIR];
  IdefixArray1D<real> y=data.x[JDIR];
  IdefixArray1D<real> xFFT("x_fft", n2fft);
  IdefixArray1D<real> Hsqr_by_rpp("Hsqr_by_rpp", n2fft);
  
  View2D<real> kernelxFFT("kernelxFFT", n1, n2fft);
  View2D<real> kernelyFFT("kernelyFFT", n1, n2fft);
  View2D<Kokkos::complex<real> > kernelxFFT_hat = myGlobals->kernelxFFT_hat;
  View2D<Kokkos::complex<real> > kernelyFFT_hat = myGlobals->kernelyFFT_hat;
  View2D<real> sigmaFFT = myGlobals->sigmaFFT;


  IdefixHostArray1D<real> xFFT_host("xFFT_host", n2fft);
  IdefixHostArray1D<real> Hsqr_by_rpp_host("Hsqr_by_rpp_host", n2fft);
  View2D_host<real> kernelxFFT_host("kernelxFFT_host", n1, n2fft);
  View2D_host<real> kernelyFFT_host("kernelyFFT_host", n1, n2fft);

  int ii = n2fft/2+1;
  real alpha = xend/xbeg;
  real ratio = pow(alpha, 1.0/n2);
  real aspect_ratio{aspect_ratio_glob};
  
  IdefixHostArray1D<real> y_host("y_host", n1);
  auto y_mirror = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(y_mirror, y);
  Kokkos::deep_copy(y_host, y_mirror);

  IdefixHostArray1D<real> dy_host("dy_host", n1);
  auto dy_mirror = Kokkos::create_mirror_view(dy);
  Kokkos::deep_copy(dy_mirror, dy);
  Kokkos::deep_copy(dy_host, dy_mirror);

  int jj = n1/2+1;
  real y_c = y_host(jj);
  real dxFFT = pow(alpha, 0.5) - pow(alpha,-0.5); 
  real L_sg;

  execution_space exec;

  for(int i = 0; i < n2fft ; i++) {
    xFFT_host(i) = (i-ii)*log(ratio);
    Hsqr_by_rpp_host(i) = pow(aspect_ratio, 2.0)*cosh(xFFT_host(i));    
  }


  for(int j = 0; j < n1 ; j++) {
    for(int i = 0; i < n2fft ; i++) {
 
      if (i==ii and j==jj){ // A fluid element does not feel it's own gravity
        kernelxFFT_host(j,i) = 0;
        kernelyFFT_host(j,i) = 0;
      } else {
        real s_sqr = 2 * (  cosh(xFFT_host(i))  -  cos(y_host(j) - y_c)  );
        real d_sqr = s_sqr / Hsqr_by_rpp_host(i);
        real ds = dxFFT * dy_host(j); 
        real X_aux = d_sqr/8.0;
  
        if (X_aux < 60) {
            L_sg = std::pow(M_PI, 0.5)
                 * X_aux
                 * std::exp(X_aux)
                 * ( std::cyl_bessel_kl(1., X_aux)
                 - std::cyl_bessel_kl(0., X_aux) );
        } else { // Taylor expansion at inifinity in order to avoid exp overflow
            L_sg = std::pow(M_PI, 0.5)
                 * X_aux
                 * 0.5 * std::pow(M_PI/2., 0.5)
                 * ( std::pow(X_aux, -1.5)
                   - 3./8.*std::pow(X_aux, -2.5)
                   + 45./128.*std::pow(X_aux, -3.5) );
        }
  
        kernelxFFT_host(j,i) = L_sg/M_PI/d_sqr * pow(exp(-xFFT_host(i))/Hsqr_by_rpp_host(i),1.5) * (exp(xFFT_host(i))-cos(y_host(j)-y_c)) * ds;
        kernelyFFT_host(j,i) = L_sg/M_PI/d_sqr * pow(exp(-xFFT_host(i))/Hsqr_by_rpp_host(i),1.5) * sin(y_host(j)-y_c) * ds;

        //if (std::isnan(kernelxFFTi_host(j, i))) { printf("NaN detected at (i=%d, j=%d)\n", i, j);}
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

  /* Fill sigmaFFT with zeros */
  idefix_for("Fill sigmaFFT with zeros", 0, n1, 0, n2fft,
              KOKKOS_LAMBDA (int j, int i) {
                sigmaFFT(j,i) = 0.0 ;
              });

  exec.fence();

  idfx::popRegion();

}



void ComputeSgForces(DataBlock &data) {
  // TODO: Check what is the value of G !
  idfx::pushRegion("SG Computation: vars");   // Profiling and debugging

  IdefixArray4D<real> Vc=data.hydro->Vc;

  int n0=data.np_tot[KDIR];
  int n1=data.np_tot[JDIR];
  int n2=data.np_tot[IDIR];
  int n2fft=2*data.np_tot[IDIR];

  View2D<Kokkos::complex<real> > kernelxFFT_hat = myGlobals->kernelxFFT_hat;
  View2D<Kokkos::complex<real> > kernelyFFT_hat = myGlobals->kernelyFFT_hat;
  IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
  IdefixArray3D<real> forceSGy = myGlobals->forceSGy;
  View2D<real> sigmaFFT = myGlobals->sigmaFFT;
  View2D<Kokkos::complex<real> > sigmaFFT_hat = myGlobals->sigmaFFT_hat;
  View2D<Kokkos::complex<real> > forceSGx_hat = myGlobals->forceSGx_hat;
  View2D<Kokkos::complex<real> > forceSGy_hat = myGlobals->forceSGy_hat;
  View2D<real> forceSGx_padded = myGlobals->forceSGx_padded;
  View2D<real> forceSGy_padded = myGlobals->forceSGy_padded;

  execution_space exec;

  idfx::popRegion();

  idfx::pushRegion("SG Computation: Plans creation");   // Profiling and debugging

  static KokkosFFT::Plan forward_plan(
        execution_space(),              // Execution space instance
        sigmaFFT,                       // Input view
        sigmaFFT_hat,                   // Output view
        KokkosFFT::Direction::forward,  // Direction
        /*axes=*/axes_type({0, 1}) 
    );

  static KokkosFFT::Plan backward_plan(
        execution_space(),              // Execution space instance
        sigmaFFT_hat,                   // Input view
        sigmaFFT,                       // Output view
        KokkosFFT::Direction::backward, // Direction
        /*axes=*/axes_type({0, 1}) 
    );

  idfx::popRegion();

  idfx::pushRegion("SG Computation: Fill sigma");   // Profiling and debugging

  idefix_for("FFT_loop", 0, n0, 0, n1, 0, n2,
              KOKKOS_LAMBDA (int k, int j, int i) {
                sigmaFFT(j,i) = Vc(RHO,k,j,i) ;
              });

  idfx::popRegion();

  idfx::pushRegion("SG Computation: FFT(sigma)");   // Profiling and debugging
  /* 2D Forward Transform */
  KokkosFFT::execute(forward_plan, sigmaFFT, sigmaFFT_hat); 

  idfx::popRegion();

  idfx::pushRegion("SG Computation: FFT(sigma)*FFT(Kernel)");   // Profiling and debugging

  /* Product in Fourier space */
  Kokkos::parallel_for(
    "FFT_loop",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n1, n2fft / 2 + 1}),
    KOKKOS_LAMBDA(int j, int i) {
        forceSGx_hat(j, i) = -kernelxFFT_hat(j, i) * sigmaFFT_hat(j, i);
        forceSGy_hat(j, i) = -kernelyFFT_hat(j, i) * sigmaFFT_hat(j, i);
    }
  ); 

  idfx::popRegion();


//  /* Shift the zero-frequency component to the center of the spectrum */
//  KokkosFFT::fftshift(exec, forceSGx_hat, 0);
//  KokkosFFT::fftshift(exec, forceSGy_hat, 0);

  idfx::pushRegion("SG Computation: FFT^-1(ForceSG)");   // Profiling and debugging

  /* 2D Backward Transform */
  KokkosFFT::execute(backward_plan, forceSGx_hat, forceSGx_padded);
  KokkosFFT::execute(backward_plan, forceSGy_hat, forceSGy_padded);

  idfx::popRegion();

//  /* Sanity check : FFT^-1(FFT(sigma)) = sigma and kernel arrays*/
//  KokkosFFT::irfft2(exec, sigmaFFT_hat, sigmaFFT);
//  idfx::DumpArray("sigmaFFT.npy", sigmaFFT); 
//
//  View2D<real> kernelxFFT("kernelxFFT", n1, n2fft);
//  View2D<real> kernelyFFT("kernelyFFT", n1, n2fft);
//  IdefixArray3D<real> kernelx_out("kernelx_out", n0, n1, n2);
//  IdefixArray3D<real> kernely_out("kernely_out", n0, n1, n2);
//
//  KokkosFFT::irfft2(exec, kernelxFFT_hat, kernelxFFT);
//  KokkosFFT::irfft2(exec, kernelyFFT_hat, kernelyFFT);
//
//  idefix_for("FFT_loop", 0, n0, 0, n1, 0, n2,
//              KOKKOS_LAMBDA (int k, int j, int i) {
//                kernelx_out(k,j,i) = kernelxFFT(j,i) ;
//                kernely_out(k,j,i) = kernelyFFT(j,i) ;
//              });
//
//  idfx::DumpArray("kernelx.npy", kernelx_out); 
//  idfx::DumpArray("kernely.npy", kernely_out); 

  exec.fence();


  idfx::pushRegion("SG Computation: Force_padded into Idefix arrays");   // Profiling and debugging

  /* Copy padded arrays into standard Idefix arrays */
  idefix_for("FFT_loop", 0, n0, 0, n1, 0, n2,
              KOKKOS_LAMBDA (int k, int j, int i) {
	        int i_offset = i+n2;
                forceSGx(k,j,i) = forceSGx_padded(j,i_offset) ;
                forceSGy(k,j,i) = forceSGy_padded(j,i_offset) ;
              });

  idfx::popRegion();
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
  IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
  IdefixArray3D<real> forceSGy = myGlobals->forceSGy;

  IdefixHostArray3D<real> forceSGx_host = variables["forceSGx"];
  IdefixHostArray3D<real> forceSGy_host = variables["forceSGy"];

  Kokkos::deep_copy(forceSGx_host, forceSGx);
  Kokkos::deep_copy(forceSGy_host, forceSGy);
}



void MySourceTerm(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;  // Main cell-centered primitive variables index
  IdefixArray4D<real> Uc = hydro->Uc;  // Main cell-centered conservative variables

  real dt = dtin;

  ComputeSgForces(*data); 
  IdefixArray3D<real> forceSGx = myGlobals->forceSGx;
  IdefixArray3D<real> forceSGy = myGlobals->forceSGy;


//  idefix_for("MySgSourceTerm",
//    0, data->np_tot[KDIR],
//    0, data->np_tot[JDIR],
//    0, data->np_tot[IDIR],
//              KOKKOS_LAMBDA (int k, int j, int i) {
//                Uc(MX1,k,j,i) += -Vc(RHO,k,j,i)*forceSGx(k,j,i)*dt;
//                Uc(MX2,k,j,i) += -Vc(RHO,k,j,i)*forceSGy(k,j,i)*dt;
//#ifndef ISOTHERMAL
//                Uc(ENG, k,j,i) += -Vc(RHO,k,j,i)*(forceSGx(k,j,i)*Vc(VX1,k,j,i) + forceSGy(k,j,i)*Vc(VX2,k,j,i)) * dt;
//#endif
//  
//  });

}



Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    // Set the function for userdefboundary
#ifdef ISOTHERMAL
    data.hydro->EnrollIsoSoundSpeed(&LISOTHSoundSpeed);
#endif
    aspect_ratio_glob = input.Get<real>("Setup","aspect_ratio",0);
    jump_radius_glob = input.Get<real>("Setup", "jump_radius",0);
    jump_width_glob = input.Get<real>("Setup", "jump_width",0);


    /* Initialise Global variables */
    myGlobals = new MyGlobalClass(data);

    if(data.haveFargo)
      data.fargo->EnrollVelocity(&FargoVelocity);

    //if(data.haveGravity)
    //  data.gravity->EnrollBodyForce(BodyForce);
    
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

    /* Initialise density */
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
      for(int j = 0; j < d.np_tot[JDIR] ; j++) {
        for(int i = 0; i < d.np_tot[IDIR] ; i++) {
          real r = d.x[IDIR](i);

          d.Vc(RHO,k,j,i) = 1.0 * pow(r, -1.5) ; 

        }
      }
    }

    // Send it all, if needed
    d.SyncToDevice();

    /* Compute self-gravity */
    ComputeSgKernel(data); 
    ComputeSgForces(data); 
    IdefixArray3D<real> forceSGx = myGlobals->forceSGx;

    /* Update velocity profile to be at centrifugal equilibrium */
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
      for(int j = 0; j < d.np_tot[JDIR] ; j++) {
        for(int i = 0; i < d.np_tot[IDIR] ; i++) {

          real r = d.x[IDIR](i);

          // correction to rotational equilibrium, taking pressure gradient into
          // account. This simplified expression was obtained with sympy;
          d.Vc(VX2,k,j,i) *= sqrt(
                               1.0 \
                               - pow(aspect_ratio, 2) / jump_width *
                                  (
                                    r * tanh((r - jump_radius) / jump_width) \
                                    - r \
                                    + 2 * jump_width
                                  ) \
 //                               - r * forceSGx_host(k,j,i) * pow(r, 0.5)
                             );

          // add some random noise to the radial velocity component break the
          // axial symmetry and let the instability grow
          //d.Vc(VX1,k,j,i) = d.Vc(VX2,k,j,i) * aspect_ratio * 1e-1*(0.5-idfx::randm());
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




