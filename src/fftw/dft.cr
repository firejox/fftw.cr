require "./libfftw.cr"
require "./types.cr"

module FFTW
  def self.dft(x : Array(Complex))
    n = x.size
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(LibFFTW::FFTWComplex).new(n)

    plan = LibFFTW.fftw_plan_dft_2d(n, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end
  end

  def self.idft(x : Array(Complex))
    n = x.size
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(LibFFTW::FFTWComplex).new(n)

    plan = LibFFTW.fftw_plan_dft_2d(n, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
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
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_3d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
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
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_3d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
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
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_4d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
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
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_4d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1]) / in_arr.size
    end.in_groups_of(n2).in_groups_of(n1)
  end

  def self.dft(x : RecArrayComplex)
    n = DimesionsHelper.dimesions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft(n.size, n, in_arr, out_slice, LibFFTW::FFTW_FORWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.ffyw_destroy_plan(plan)

    DimensionsHelper.build_array_by_dimesions(n.each.skip(1), out_slice.map { |elem| Complex.new(elem[0], elem[1]) })
  end

  def self.idft(x : RecArrayComplex)
    n = DimesionsHelper.dimesions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft(n.size, n, in_arr, out_slice, LibFFTW::FFTW_BACKWARD, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    DimensionsHelper.build_array_by_dimesions(n.each.skip(1), out_slice.map { |elem| Complex.new(elem[0], elem[1]) / in_arr.size })
  end

  def self.dft(x : Array(Float64))
    n = x.size
    out_slice = Slice(LibFFTW::FFTWComplex).new(n)

    plan = LibFFTW.fftw_plan_dft_r2c_1d(n, x, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end
  end

  def self.idft_r(x : Array(Complex))
    n = x.size
    in_arr = x.map { |elem| StaticArray[elem.real, elem.imag] } 
    out_slice = Slice(Float64).new(n)

    plan = LibFFTW.fftw_plan_dft_c2r_1d(n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      elem / n
    end
  end

  def self.dft(x : Array(Array(Float64)))
    n0 = x.size
    n1 = x[0].size
    in_arr = x.flatten
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_r2c_2d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end.in_groups_of(n1)
  end

  def self.idft_r(x : Array(Array(Complex)))
    n0 = x.size
    n1 = x[0].size
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_c2r_2d(n0, n1, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      elem / in_arr.size
    end.in_groups_of(n1)
  end

  def self.dft(x : Array(Array(Array(Float64))))
    n0 = x.size
    n1 = x[0].size
    n2 = x[0][0].size
    in_arr = x.flatten
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_r2c_3d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      Complex.new(elem[0], elem[1])
    end.in_groups_of(n2).in_groups_of(n1)
  end

  def self.idft_r(x : Array(Array(Array(Complex)))))
    n0 = x.size
    n1 = x[0].size
    n2 = s[0][0].size
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_c2r_3d(n0, n1, n2, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)

    out_slice.map do |elem|
      elem / in_arr.size
    end.in_groups_of(n2).in_groups_of(n1)
  end

  def self.dft(x : RecArrayFloat64)
    n = DimensionsHelper.dimensions(x)
    in_arr = x.flatten
    out_slice = Slice(LibFFTW::FFTWComplex).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_r2c(n.size, n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.ffyw_destroy_plan(plan)

    DimensionsHelper.build_array_by_dimesions(n.each.skip(1), out_slice.map { |elem| Complex.new(elem[0], elem[1]) })
  end

  def self.idft_r(x : RecArrayComplex)
    n = DimesionsHelper.dimesions(x)
    in_arr = x.flatten.map { |elem| StaticArray[elem.real, elem.imag] }
    out_slice = Slice(Float64).new(in_arr.size)

    plan = LibFFTW.fftw_plan_dft_r2c(n.size, n, in_arr, out_slice, LibFFTW::FFTW_ESTIMATE)
    LibFFTW.fftw_execute(plan)
    LibFFTW.ffyw_destroy_plan(plan)

    DimensionsHelper.build_array_by_dimesions(n.each.skip(1), out_slice.map { |elem| elem / in_arr.size })
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

  def self.build_array_by_dimesions(iter, ary)
    a = iter.next
    if a == Iterator.stop
      ary
    else
      build_array_by_dimensions(iter, ary).in_groups_of(a.as(Int32))
    end
  end
end
