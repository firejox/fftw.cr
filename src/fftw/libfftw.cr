module FFTW
  @[Link("fftw3")]
  @[Link("m")]
  lib LibFFTW
    alias FFTWComplex = Float64[2]

    alias FFTWPlan = Void*

    enum FFTWR2RKind
      FFTW_R2HC    =  0
      FFTW_HC2R    =  1
      FFTW_DHT     =  2
      FFTW_REDFT00 =  3
      FFTW_REDFT01 =  4
      FFTW_REDFT10 =  5
      FFTW_REDFT11 =  6
      FFTW_RODFT00 =  7
      FFTW_RODFT01 =  8
      FFTW_RODFT10 =  9
      FFTW_RODFT11 = 10
    end

    fun fftw_plan_dft_1d(n : Int32, in_arr : FFTWComplex*, out_arr : FFTWComplex*, sign : Int32, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_2d(n0 : Int32, n1 : Int32, in_arr : FFTWComplex*, out_arr : FFTWComplex*, sign : Int32, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_3d(n0 : Int32, n1 : Int32, n2 : Int32, in_arr : FFTWComplex*, out_arr : FFTWComplex*, sign : Int32, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft(rank : Int32, n : Int32*, in_arr : FFTWComplex*, out_arr : FFTWComplex*, sign : Int32, flags : UInt32) : FFTWPlan

    fun fftw_plan_dft_r2c_1d(n : Int32, in_arr : Float64*, out_arr : FFTWComplex*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_c2r_1d(n : Int32, in_arr : FFTWComplex*, out_arr : Float64*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_r2c_2d(n0 : Int32, n1 : Int32, in_arr : Float64*, out_arr : FFTWComplex*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_c2r_2d(n0 : Int32, n1 : Int32, in_arr : FFTWComplex*, out_arr : Float64*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_r2c_3d(n : Int32, n1 : Int32, n2 : Int32, in_arr : Float64*, out_arr : FFTWComplex*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_c2r_3d(n : Int32, n1 : Int32, n2 : Int32, in_arr : FFTWComplex*, out_arr : Float64*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_r2c(rank : Int32, n : Int32*, in_arr : Float64*, out_arr : FFTWComplex*, flags : UInt32) : FFTWPlan
    fun fftw_plan_dft_c2r(rank : Int32, n : Int32*, in_arr : FFTWComplex*, out_arr : Float64*, flags : UInt32) : FFTWPlan

    fun fftw_plan_r2r_1d(n : Int32, in_arr : Float64*, out_arr : Float64*, kind : FFTWR2RKind, flags : UInt32) : FFTWPlan
    fun fftw_plan_r2r_2d(n0 : Int32, n1 : Int32, in_arr : Float64*, out_arr : Float64*, kind : FFTWR2RKind, flags : UInt32) : FFTWPlan
    fun fftw_plan_r2r_3d(n0 : Int32, n1 : Int32, n2 : Int32, in_arr : Float64*, out_arr : Float64*, kind : FFTWR2RKind, flags : UInt32) : FFTWPlan
    fun fftw_plan_r2r(rank : Int32, n : Int32*, in_arr : Float64*, out_arr : Float64*, kind : FFTWR2RKind, flags : UInt32) : FFTWPlan

    fun fftw_execute(plan : FFTWPlan) : Void
    fun fftw_destroy_plan(plan : FFTWPlan) : Void

    fun fftw_cleanup : Void

    FFTW_FORWARD  = -1
    FFTW_BACKWARD =  1

    FFTW_NO_TIME_LINIT = -1.0

    FFTW_MEASURE         = 0b0
    FFTW_DESTROY_INPUT   = 0b1
    FFTW_UNALIGEND       = 0b1 << 1
    FFTW_CONSERVE_MEMORY = 0b1 << 2
    FFTW_EXHAUSIVE       = 0b1 << 3
    FFTW_PRESERVE_INPUT  = 0b1 << 4
    FFTW_PATIENT         = 0b1 << 5
    FFTW_ESTIMATE        = 0b1 << 6
    FFTW_WISDOM_ONLY     = 0b1 << 21
  end
end
