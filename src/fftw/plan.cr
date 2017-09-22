require "./libfftw.cr"

module FFTW
  class Plan
    @c_slice_in : Slice(LibFFTW::FFTWComplex)
    @c_slice_out : Slice(LibFFTW::FFTWComplex)

    def initialize(n : Int32)
      @r_slice_in = Slice(Float64).new(n)
      @r_slice_out = Slice(Float64).new(n)
      @c_slice_in = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)
      @c_slice_out = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)

      @dft_c2c_plan = LibFFTW.fftw_plan_dft_1d(n, @c_slice_in, @c_slice_out, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_MEASURE)
      @idft_c2c_plan = LibFFTW.fftw_plan_dft_1d(n, @c_slice_in, @c_slice_out, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_MEASURE)
      @dft_r2c_plan = LibFFTW.fftw_plan_dft_r2c_1d(n, @r_slice_in, @c_slice_out, LibFFTW::FFTW_MEASURE)
      @idft_c2r_plan = LibFFTW.fftw_plan_dft_c2r_1d(n, @c_slice_in, @r_slice_out, LibFFTW::FFTW_MEASURE)

      @dimensions = [n]
      @rc_dimensions = [(n/2) + 1]
    end

    def initialize(n0 : Int32, n1 : Int32)
      n = n0 * n1
      @r_slice_in = Slice(Float64).new(n)
      @r_slice_out = Slice(Float64).new(n)
      @c_slice_in = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)
      @c_slice_out = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)

      @dft_c2c_plan = LibFFTW.fftw_plan_dft_2d(n0, n1, @c_slice_in, @c_slice_out, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_MEASURE)
      @idft_c2c_plan = LibFFTW.fftw_plan_dft_2d(n0, n1, @c_slice_in, @c_slice_out, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_MEASURE)
      @dft_r2c_plan = LibFFTW.fftw_plan_dft_r2c_2d(n0, n1, @r_slice_in, @c_slice_out, LibFFTW::FFTW_MEASURE)
      @idft_c2r_plan = LibFFTW.fftw_plan_dft_c2r_2d(n0, n1, @c_slice_in, @r_slice_out, LibFFTW::FFTW_MEASURE)

      @dimensions = [n0, n1]
      @rc_dimensions = [n0, (n1/2) + 1]
    end

    def initialize(n0 : Int32, n1 : Int32, n2 : Int32)
      n = n0 * n1 * n2
      @r_slice_in = Slice(Float64).new(n)
      @r_slice_out = Slice(Float64).new(n)
      @c_slice_in = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)
      @c_slice_out = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)

      @dft_c2c_plan = LibFFTW.fftw_plan_dft_3d(n0, n1, n2, @c_slice_in, @c_slice_out, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_MEASURE)
      @idft_c2c_plan = LibFFTW.fftw_plan_dft_3d(n0, n1, n2, @c_slice_in, @c_slice_out, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_MEASURE)
      @dft_r2c_plan = LibFFTW.fftw_plan_dft_r2c_3d(n0, n1, n2, @r_slice_in, @c_slice_out, LibFFTW::FFTW_MEASURE)
      @idft_c2r_plan = LibFFTW.fftw_plan_dft_c2r_3d(n0, n1, n2, @c_slice_in, @r_slice_out, LibFFTW::FFTW_MEASURE)

      @dimensions = [n0, n1, n2]
      @rc_dimensions = [n0, n1, (n2/2) + 1]
    end

    def initialize(*dimensions : Int32)
      @dimensions = dimensions.to_a
      @rc_dimensions = @dimensions.dup
      @rc_dimensions[-1] = (@rc_dimensions[-1] / 2) + 1
      n = dimensions.product { |x| x }
      @r_slice_in = Slice(Float64).new(n)
      @r_slice_out = Slice(Float64).new(n)
      @c_slice_in = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)
      @c_slice_out = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)

      @dft_c2c_plan = LibFFTW.fftw_plan_dft(dimensions.size, @dimensions, @c_slice_in, @c_slice_out, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_MEASURE)
      @idft_c2c_plan = LibFFTW.fftw_plan_dft(dimensions.size, @dimensions, @c_slice_in, @c_slice_out, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_MEASURE)
      @dft_r2c_plan = LibFFTW.fftw_plan_dft_r2c(dimensions.size, @dimensions, @r_slice_in, @c_slice_out, LibFFTW::FFTW_MEASURE)
      @idft_c2r_plan = LibFFTW.fftw_plan_dft_c2r(dimensions.size, @dimensions, @c_slice_in, @r_slice_out, LibFFTW::FFTW_MEASURE)
    end

    def dft(x)
      in_arr = x.flatten
      dim = nil
      iter = nil

      try_complex(typeof(in_arr[0])) do
        @c_slice_in.size.times do |i|
          @c_slice_in[i] = StaticArray[in_arr[i].real, in_arr[i].imag]
        end

        LibFFTW.fftw_execute(@dft_c2c_plan)

        iter = @c_slice_out.map { |elem| Complex.new(elem[0], elem[1]) }.each
        dim = @dimensions
      end

      try_f64(typeof(in_arr[0])) do
        @r_slice_in.copy_from(in_arr.to_unsafe, @r_slice_in.size)

        LibFFTW.fftw_execute(@dft_r2c_plan)

        sz = @r_slice_in.size / @dimensions[-1] * (@rc_dimensions[-1])
        iter = @c_slice_out[0, sz].map { |elem| Complex.new(elem[0], elem[1]) }.each

        dim = @rc_dimensions
      end

      return DimensionsHelper.build_array_by_dimensions(typeof(x), dim.not_nil!, 0) { iter.not_nil!.next.as(Complex) }
    end

    def idft(x)
      in_arr = x.flatten
      n = @c_slice_in.size
      n.times do |i|
        @c_slice_in[i] = StaticArray[in_arr[i].real, in_arr[i].imag]
      end

      LibFFTW.fftw_execute(@idft_c2c_plan)

      iter = @c_slice_out.map { |elem| Complex.new(elem[0], elem[1]) / n }.each

      DimensionsHelper.build_array_by_dimensions(typeof(x), @dimensions, 0) { iter.next.as(Complex) }
    end

    def idft_r(x)
      in_arr = x.flatten
      n = @c_slice_in.size
      sz = n / @dimensions[-1] * @rc_dimensions[-1]
      sz.times do |i|
        @c_slice_in[i] = StaticArray[in_arr[i].real, in_arr[i].imag]
      end

      LibFFTW.fftw_execute(@idft_c2r_plan)

      iter = @r_slice_out.map { |elem| elem / n }.each

      DimensionsHelper.build_array_by_dimensions(typeof(x), @dimensions, 0) { iter.next.as(Float64) }
    end

    private def try_complex(type : T.class, &block) forall T
      {% if T == Complex %}
        yield
      {% end %}
    end

    private def try_f64(type : T.class, &block) forall T
      {% if T == Float64 %}
        yield
      {% end %}
    end

    def finalize
      LibFFTW.fftw_destroy_plan(@dft_c2c_plan)
      LibFFTW.fftw_destroy_plan(@idft_c2c_plan)
      LibFFTW.fftw_destroy_plan(@dft_r2c_plan)
      LibFFTW.fftw_destroy_plan(@idft_c2r_plan)
    end
  end
end

private struct DimensionsHelper
  def self.dimensions(ary)
    result = [] of Int32
    dimensions ary, result
    result
  end

  def self.dimensions(ary, result)
    result << ary.size
    if ary[0].is_a?(Array)
      dimensions ary[0], result
    end
  end

  def self.build_array_by_dimensions(type : Array(T).class, n, depth, &block) forall T
    Array.new(n[depth]) { build_array_by_dimensions(T, n, depth + 1) { yield } }
  end

  def self.build_array_by_dimensions(type : T.class, n, depth, &block) forall T
    yield
  end
end
