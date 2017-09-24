require "./libfftw.cr"

module FFTW
  abstract class Plan
    def self.new(*dims : *T) forall T
      {% if T.size == 0 %}
        {{ raise "Rank must not be zero" }}
      {% end %}
      {{ "FFTWPlanImpl(#{T.size}).new(*dims)".id }}
    end

    abstract def dft(x)
    abstract def idft(x)
  end
end

private class FFTWPlanImpl(N) < FFTW::Plan
  @c_slice_in : Slice(FFTW::LibFFTW::FFTWComplex)
  @c_slice_out : Slice(FFTW::LibFFTW::FFTWComplex)

  @dft_c2c_plan = Pointer(Void).null
  @idft_c2c_plan = Pointer(Void).null

  def initialize(*dims : Int32)
    n = dims.product { |x| x }

    @c_slice_in = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)
    @c_slice_out = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    @dft_c2c_plan = FFTW::LibFFTW.fftw_plan_dft(N, pointerof(dims).as(Int32*), @c_slice_in, @c_slice_out, FFTW::LibFFTW::FFTW_FORWARD, FFTW::LibFFTW::FFTW_MEASURE)
    @idft_c2c_plan = FFTW::LibFFTW.fftw_plan_dft(N, pointerof(dims).as(Int32*), @c_slice_in, @c_slice_out, FFTW::LibFFTW::FFTW_BACKWARD, FFTW::LibFFTW::FFTW_MEASURE)
  end

  def dft(x : Array(Complex))
    @c_slice_in.size.times do |i|
      @c_slice_in[i] = StaticArray[x[i].real, x[i].imag]
    end

    FFTW::LibFFTW.fftw_execute(@dft_c2c_plan)

    @c_slice_out.map { |elem| Complex.new(elem[0], elem[1]) }
  end

  def idft(x : Array(Complex))
    @c_slice_in.size.times do |i|
      @c_slice_in[i] = StaticArray[x[i].real, x[i].imag]
    end

    FFTW::LibFFTW.fftw_execute(@idft_c2c_plan)

    @c_slice_out.map { |elem| Complex.new(elem[0], elem[1]) }
  end

  def finalize
    FFTW::LibFFTW.fftw_destroy_plan(@dft_c2c_plan)
    FFTW::LibFFTW.fftw_destroy_plan(@idft_c2c_plan)
  end
end
