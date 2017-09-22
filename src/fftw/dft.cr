require "./libfftw.cr"
require "./types.cr"

module FFTW
  def self.dft(x : Array(Complex))
    n = x.size
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    plan = LibFFTW.fftw_plan_dft_1d(n, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end
  end

  def self.idft(x : Array(Complex))
    n = x.size
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(n).to_slice(n)

    plan = LibFFTW.fftw_plan_dft_1d(n, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1]) / n
    end
  end

  def self.dft(x : Array(Array(Complex)))
    n0 = x.size
    n1 = x[0].size
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_2d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end.in_groups_of(n1)
  end

  def self.idft(x : Array(Array(Complex)))
    n0 = x.size
    n1 = x[0].size
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_2d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1]) / in_arr.size
    end.in_groups_of(n1)
  end

  def self.dft(x : Array(Array(Array(Complex))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_3d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end.in_groups_of(n2).in_groups_of(n1)
  end

  def self.idft(x : Array(Array(Array(Complex))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_3d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1]) / in_arr.size
    end.in_groups_of(n2).in_groups_of(n1)
  end

  def self.dft(x : RecArrayComplex)
    n = DimesionsHelper.dimesions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft(n.size, n, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.ffyw_destroy_plan(plan)

    iter = out_slice.map { |elem| Complex.new(elem[0], elem[1]) }.each
    DimensionsHelper.build_array_by_dimesions(typeof(x), n, 0) { iter.next.as(Complex) }
  end

  def self.idft(x : RecArrayComplex)
    n = DimesionsHelper.dimesions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(in_arr.size).to_slice(in_arr.size)

    plan = LibFFTW.fftw_plan_dft(n.size, n, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    iter = out_slice.map { |elem| Complex.new(elem[0], elem[1]) / in_arr.size }.each

    DimensionsHelper.build_array_by_dimesions(typeof(x), n, 0) { iter.next.as(Complex) }
  end

  def self.dft(x : Array(Float64))
    sz = x.size/2 + 1
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(sz).to_slice(sz)

    plan = LibFFTW.fftw_plan_dft_r2c_1d(x.size, x, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end
  end

  def self.idft_r(x : Array(Complex), sz : Int32)
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(sz)

    plan = LibFFTW.fftw_plan_dft_c2r_1d(sz, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      elem / sz
    end
  end

  def self.dft(x : Array(Array(Float64)))
    n0 = x.size
    n1 = x[0].size
    in_arr = x.flatten
    sz = n0 * (n1/2 + 1)
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(sz).to_slice(sz)

    plan = LibFFTW.fftw_plan_dft_r2c_2d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end.in_groups_of(n1/2 + 1)
  end

  def self.idft_r(x : Array(Array(Complex)), last_dim : Int32)
    n0 = x.size
    sz = last_dim * n0

    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(sz)

    plan = LibFFTW.fftw_plan_dft_c2r_2d(n0, last_dim, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      elem / sz
    end.in_groups_of(last_dim)
  end

  def self.dft(x : Array(Array(Array(Float64))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size
    sz = n0 * n1 * (n2/2 + 1)

    in_arr = x.flatten
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(sz).to_slice(sz)

    plan = LibFFTW.fftw_plan_dft_r2c_3d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end.in_groups_of(n2/2 + 1).in_groups_of(n1)
  end

  def self.idft_r(x : Array(Array(Array(Complex))), last_dim : Int32)
    n0 = x.size
    n1 = x[0].size
    n = n0 * n1 * last_dim
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(n)

    plan = LibFFTW.fftw_plan_dft_c2r_3d(n0, n1, last_dim, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      elem / n
    end.in_groups_of(last_dim).in_groups_of(n1)
  end

  def self.dft(x : RecArrayFloat64)
    n = DimensionsHelper.dimensions(x)
    in_arr = x.flatten
    sz = in_arr.sz / n[-1] * ((n[-1] / 2) + 1)
    out_slice = Pointer(LibFFTW::FFTWComplex).malloc(sz).to_slice(sz)

    plan = LibFFTW.fftw_plan_dft_r2c(n.size, n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.ffyw_destroy_plan(plan)

    iter = out_slice.map { |elem| Complex.new(elem[0], elem[1]) }.each
    n[-1] = (n[-1] / 2) + 1

    DimensionsHelper.build_array_by_dimesions(typeof(x), n, 0) { iter.next.as(Complex) }
  end

  def self.idft_r(x : RecArrayFloat64, last_dim : Int32)
    n = DimesionsHelper.dimesions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    sz = in_arr.size / n[-1] * last_dim
    n[-1] = last_dim
    out_slice = Slice(Float64).new(sz)

    plan = LibFFTW.fftw_plan_dft_r2c(n.size, n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.ffyw_destroy_plan(plan)

    iter = out_slice.map { |elem| elem / sz }.each

    DimensionsHelper.build_array_by_dimesions(typeof(x), n, 0) { iter.next.as(Float64) }
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
    if ary[1].is_a?(Array)
      dimensions ary[1], result
    end
  end

  def self.build_array_by_dimensions(type : Array(T).class, n, depth, &block) forall T
    Array.new(n[depth]) { build_array_by_dimensions(T, n, depth + 1) { yield } }
  end

  def self.build_array_by_dimensions(type : T.class, n, depth, &block) forall T
    yield
  end
end
