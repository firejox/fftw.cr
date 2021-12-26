require "./libfftw.cr"

module FFTW
  abstract class Plan
    def self.new(*dims : *T) forall T
      {% if T.size == 0 %}
        {{ raise "Rank must not be zero" }}
      {% end %}
      {{ "FFTWPlanImpl(#{T.size}).new(*dims)".id }}
    end

    abstract def dft(x : Array(Complex))
    abstract def idft(x : Array(Complex))

    abstract def dft(x : Array(Float64))
    abstract def idft_r(x : Array(Complex))
  end
end

private class FFTWPlanImpl(N) < FFTW::Plan
  @c_slice_in : Slice(FFTW::LibFFTW::FFTWComplex)
  @c_slice_out : Slice(FFTW::LibFFTW::FFTWComplex)

  @dft_c2c_plan = Pointer(Void).null
  @idft_c2c_plan = Pointer(Void).null

  @dft_r2c_plan = Pointer(Void).null
  @idft_c2r_plan = Pointer(Void).null

  @r2c_size : Int32

  def initialize(*dims : Int32)
    n = dims.product { |x| x }

    @c_slice_in = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)
    @c_slice_out = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    @r_slice_in = Slice(Float64).new(n)
    @r_slice_out = Slice(Float64).new(n)
    @r2c_size = n // dims.last * (dims.last//2 + 1)

    @dft_c2c_plan = FFTW::LibFFTW.fftw_plan_dft(N, pointerof(dims).as(Int32*), @c_slice_in, @c_slice_out, FFTW::LibFFTW::FFTW_FORWARD, FFTW::LibFFTW::FFTW_MEASURE)
    @idft_c2c_plan = FFTW::LibFFTW.fftw_plan_dft(N, pointerof(dims).as(Int32*), @c_slice_in, @c_slice_out, FFTW::LibFFTW::FFTW_BACKWARD, FFTW::LibFFTW::FFTW_MEASURE)

    @dft_r2c_plan = FFTW::LibFFTW.fftw_plan_dft_r2c(N, pointerof(dims).as(Int32*), @r_slice_in, @c_slice_out, FFTW::LibFFTW::FFTW_MEASURE)
    @idft_c2r_plan = FFTW::LibFFTW.fftw_plan_dft_c2r(N, pointerof(dims).as(Int32*), @c_slice_in, @r_slice_out, FFTW::LibFFTW::FFTW_MEASURE)
  end

  def dft(x : Array(Complex)) : Array(Complex)
    @c_slice_in.size.times do |i|
      @c_slice_in[i] = StaticArray[x[i].real, x[i].imag]
    end

    FFTW::LibFFTW.fftw_execute(@dft_c2c_plan)

    Array(Complex).new(@c_slice_out.size) do |i|
      elem = @c_slice_out[i]
      Complex.new(elem[0], elem[1])
    end
  end

  def idft(x : Array(Complex))
    @c_slice_in.size.times do |i|
      @c_slice_in[i] = StaticArray[x[i].real, x[i].imag]
    end

    FFTW::LibFFTW.fftw_execute(@idft_c2c_plan)

    @c_slice_out.map { |elem| Complex.new(elem[0], elem[1]) }
  end

  def dft(x : Array(Float64)) : Array(Complex)
    @r_slice_in.copy_from(x.to_unsafe, @r_slice_in.size)

    FFTW::LibFFTW.fftw_execute(@dft_r2c_plan)

    Array(Complex).new(@r2c_size) do |i|
      elem = @c_slice_out[i]
      Complex.new(elem[0], elem[1])
    end
  end

  def idft_r(x : Array(Complex))
    @r2c_size.times do |i|
      @c_slice_in[i] = StaticArray[x[i].real, x[i].imag]
    end

    FFTW::LibFFTW.fftw_execute(@idft_c2r_plan)

    @r_slice_out.to_a
  end

  def finalize
    FFTW::LibFFTW.fftw_destroy_plan(@dft_c2c_plan)
    FFTW::LibFFTW.fftw_destroy_plan(@idft_c2c_plan)

    FFTW::LibFFTW.fftw_destroy_plan(@dft_r2c_plan)
    FFTW::LibFFTW.fftw_destroy_plan(@idft_c2r_plan)
  end
end
