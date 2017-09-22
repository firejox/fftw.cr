require "./spec_helper"

r = Random.new

describe FFTW do
  it "can compute abitrary 1d dft" do
    arr = Array.new(32) { Complex.new(r.next_u, r.next_u) }
    dft_arr_fallback = Array.new(32) do |i|
      x = Complex.new(0, 0)
      arr.each_with_index do |a, j|
        x += a * Complex.new(Math.cos(Math::PI * 2.0 * i * j / 32.0), -Math.sin(Math::PI * 2.0 * i * j / 32.0))
      end
      x
    end

    dft_arr = FFTW.dft(arr)

    dft_arr.zip(dft_arr_fallback) do |a, b|
      a.should be_close(b, 1e-2)
    end
  end

  it "can compute abitary 1d inverse dft" do
    arr = Array.new(32) { Complex.new(r.next_u, r.next_u) }

    arr_back = FFTW.idft(FFTW.dft(arr))

    arr_back.zip(arr) do |a, b|
      a.should be_close(b, 1e-2)
    end
  end
end
