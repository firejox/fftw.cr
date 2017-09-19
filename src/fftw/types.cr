require "complex"

module FFTW
  alias RecArrayComplex = Array(Array(Array(Array(Complex)))) | Array(RecArrayComplex)
  alias RecArrayFloat64 = Array(Array(Array(Array(Float64)))) | Array(RecArrayFloat64)
end
