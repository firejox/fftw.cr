require "./libfftw.cr"
require "./types.cr"

module FFTW
  def self.dft(*args : *T) forall T
    {{ "DFTCompute(#{T.size - 1}).dft(*args)".id }}
  end

  def self.idft(*args : *T) forall T
    {{ "DFTCompute(#{T.size - 1}).idft(*args)".id }}
  end

  def self.idft_r(*args : *T) forall T
    {{ "DFTCompute(#{T.size - 1}).idft_r(*args)".id }}
  end

  def self.dft(x : Array(Complex))
    DFTCompute(1).dft(x, x.size)
  end

  def self.idft(x : Array(Complex))
    DFTCompute(1).idft(x, x.size)
  end

  def self.dft(x : Array(Array(Complex)))
    n0 = x.size
    n1 = x[0].size

    iter = DFTCompute(2).dft(x.flatten, n0, n1).each

    Array.new(n0) do
      Array.new(n1) do
        iter.next.as(Complex)
      end
    end
  end

  def self.idft(x : Array(Array(Complex)))
    n0 = x.size
    n1 = x[0].size

    iter = DFTCompute(2).idft(x.flatten, n0, n1).each

    Array.new(n0) do
      Array.new(n1) do
        iter.next.as(Complex)
      end
    end
  end

  def self.dft(x : Array(Array(Array(Complex))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size

    iter = DFTCompute(3).dft(x.flatten, n0, n1, n2).each

    Array.new(n0) do
      Array.new(n1) do
        Array.new(n2) do
          iter.next.as(Complex)
        end
      end
    end
  end

  def self.idft(x : Array(Array(Array(Complex))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size

    iter = DFTCompute(3).idft(x.flatten, n0, n1, n2).each

    Array.new(n0) do
      Array.new(n1) do
        Array.new(n2) do
          iter.next.as(Complex)
        end
      end
    end
  end

  def self.dft(x : RecArrayComplex)
    n = DimensionsHelper.dimensions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft(n.size, n, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    iter = out_slice.map { |elem| Complex.new(elem[0], elem[1]) }.each
    DimensionsHelper.build_array_by_dimensions(typeof(x), n, 0) { iter.next.as(Complex) }
  end

  def self.idft(x : RecArrayComplex)
    n = DimensionsHelper.dimensions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    sz = in_arr.size
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(sz).to_slice(sz)

    plan = LibFFTW.fftw_plan_dft(n.size, n, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    iter = out_slice.map { |elem| Complex.new(elem[0], elem[1]) }.each

    DimensionsHelper.build_array_by_dimensions(typeof(x), n, 0) { iter.next.as(Complex) }
  end

  def self.dft(x : Array(Float64))
    DFTCompute(1).dft(x, x.size)
  end

  def self.idft_r(x : Array(Complex), sz : Int32)
    DFTCompute(1).idft_r(x, sz)
  end

  def self.dft(x : Array(Array(Float64)))
    n0 = x.size
    n1 = x[0].size

    iter = DFTCompute(2).dft(x.flatten, n0, n1).each

    Array.new(n0) do
      Array.new((n1//2) + 1) do
        iter.next.as(Complex)
      end
    end
  end

  def self.idft_r(x : Array(Array(Complex)), last_dim : Int32)
    n0 = x.size

    iter = DFTCompute(2).idft_r(x.flatten, n0, last_dim).each

    Array.new(n0) do
      Array.new(last_dim) do
        iter.next.as(Float64)
      end
    end
  end

  def self.dft(x : Array(Array(Array(Float64))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size

    iter = DFTCompute(3).dft(x.flatten, n0, n1, n2).each

    Array.new(n0) do
      Array.new(n1) do
        Array.new((n2//2) + 1) do
          iter.next.as(Complex)
        end
      end
    end
  end

  def self.idft_r(x : Array(Array(Array(Complex))), last_dim : Int32)
    n0 = x.size
    n1 = x[0].size

    iter = DFTCompute(3).idft_r(x.flatten, n0, n1, last_dim).each

    Array.new(n0) do
      Array.new(n1) do
        Array.new(last_dim) do
          iter.next.as(Float64)
        end
      end
    end
  end

  def self.dft(x : RecArrayFloat64)
    n = DimensionsHelper.dimensions(x)
    in_arr = x.flatten
    sz = in_arr.size // n[-1] * ((n[-1] // 2) + 1)
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(sz).to_slice(sz)

    plan = LibFFTW.fftw_plan_dft_r2c(n.size, n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    iter = out_slice.map { |elem| Complex.new(elem[0], elem[1]) }.each
    n[-1] = (n[-1] // 2) + 1

    DimensionsHelper.build_array_by_dimensions(typeof(x), n, 0) { iter.next.as(Complex) }
  end

  def self.idft_r(x : RecArrayComplex, last_dim : Int32)
    n = DimensionsHelper.dimensions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    sz = in_arr.size // n[-1] * last_dim
    n[-1] = last_dim
    out_slice = Slice(Float64).new(sz)

    plan = LibFFTW.fftw_plan_dft_c2r(n.size, n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    iter = out_slice.map { |elem| elem }.each

    DimensionsHelper.build_array_by_dimensions(typeof(x), n, 0) { iter.next.as(Float64) }
  end
end

private struct DimensionsHelper
  def self.dimensions(ary)
    result = [] of Int32
    dimensions ary, result
    result
  end

  def self.dimensions(ary, result)
    if ary.is_a?(Array)
      result << ary.size
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

private struct DFTCompute(N)
  def self.dft(x : Array(Complex), *dims : Int32)
    n = dims.product { |x| x }
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    plan = FFTW::LibFFTW.fftw_plan_dft(N, pointerof(dims).as(Int32*), in_arr, out_slice, FFTW::LibFFTW::FFTW_FORWARD, FFTW::LibFFTW::FFTW_ESTIMATE)

    FFTW::LibFFTW.fftw_execute(plan)
    FFTW::LibFFTW.fftw_destroy_plan(plan)

    Array(Complex).new(out_slice.size) do |i|
      elem = out_slice[i]
      Complex.new(elem[0], elem[1])
    end
  end

  def self.idft(x : Array(Complex), *dims : Int32)
    n = dims.product { |x| x }
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    plan = FFTW::LibFFTW.fftw_plan_dft(N, pointerof(dims).as(Int32*), in_arr, out_slice, FFTW::LibFFTW::FFTW_BACKWARD, FFTW::LibFFTW::FFTW_ESTIMATE)

    FFTW::LibFFTW.fftw_execute(plan)
    FFTW::LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end
  end

  def self.dft(x : Array(Float), *dims : Int32)
    n = dims.product { |x| x } // dims.last * ((dims.last // 2) + 1)
    out_slice = Pointer(FFTW::LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    plan = FFTW::LibFFTW.fftw_plan_dft_r2c(N, pointerof(dims).as(Int32*), x, out_slice, FFTW::LibFFTW::FFTW_ESTIMATE)
    FFTW::LibFFTW.fftw_execute(plan)
    FFTW::LibFFTW.fftw_destroy_plan(plan)

    Array(Complex).new(out_slice.size) do |i|
      elem = out_slice[i]
      Complex.new(elem[0], elem[1])
    end
  end

  def self.idft_r(x : Array(Complex), *dims : Int32)
    n = dims.product { |x| x }
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(n)

    plan = FFTW::LibFFTW.fftw_plan_dft_c2r(N, pointerof(dims).as(Int32*), in_arr, out_slice, FFTW::LibFFTW::FFTW_ESTIMATE)

    FFTW::LibFFTW.fftw_execute(plan)
    FFTW::LibFFTW.fftw_destroy_plan(plan)

    out_slice.to_a
  end
end
