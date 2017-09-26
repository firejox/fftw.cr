require "./spec_helper"

r = Random.new

describe FFTW do
  describe "DFT of Complex Data" do
    it "can compute abitrary size of 1d DFT" do
      arr = Array.new(32) { Complex.new(r.next_u, r.next_u) }
      dft_arr_fallback = Array.new(32) do |i|
        x = Complex.new(0, 0)
        arr.each_with_index do |a, j|
          x += a * Complex.new(Math.cos(Math::PI * i * j / 16.0), -Math.sin(Math::PI * i * j / 16.0))
        end
        x
      end

      dft_arr = FFTW.dft(arr)

      dft_arr.zip(dft_arr_fallback) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitary size of 1d inverse DFT" do
      arr = Array.new(32) { Complex.new(r.next_u, r.next_u) }

      arr_back = FFTW.idft(FFTW.dft(arr))

      arr_back.zip(arr) do |a, b|
        a.should be_close(b * 32, 1e-2)
      end
    end

    it "can compute abitrary size of 2d DFT" do
      mat = Array.new(8) { Array.new(8) { Complex.new(r.next_u, r.next_u) } }
      dft_mat_fallback = Array.new(8) do |i|
        Array.new(8) do |j|
          x = Complex.new(0, 0)
          mat.each_with_index do |ary, k|
            ary.each_with_index do |y, l|
              c = i * k / 4.0 + j * l / 4.0
              x += y * Complex.new(Math.cos(Math::PI * c), -Math.sin(Math::PI * c))
            end
          end
          x
        end
      end

      dft_mat = FFTW.dft(mat)

      dft_mat.flatten.zip(dft_mat_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitrary size of 2d inverse DFT" do
      mat = Array.new(8) { Array.new(8) { Complex.new(r.next_u, r.next_u) } }

      mat_back = FFTW.idft(FFTW.dft(mat))

      mat_back.flatten.zip(mat.flatten) do |a, b|
        a.should be_close(b * 64, 1e-2)
      end
    end

    it "can compute abitrary size of 3d DFT" do
      cube = Array.new(8) { Array.new(8) { Array.new(8) { Complex.new(r.next_u, r.next_u) } } }
      dft_cube_fallback = Array.new(8) do |i1|
        Array.new(8) do |i2|
          Array.new(8) do |i3|
            x = Complex.new(0, 0)

            cube.each_with_index do |mat, j1|
              mat.each_with_index do |ary, j2|
                ary.each_with_index do |y, j3|
                  c = (i1 * j1 + i2 * j2 + i3 * j3) / 4.0

                  x += y * Complex.new(Math.cos(Math::PI * c), -Math.sin(Math::PI * c))
                end
              end
            end
            x
          end
        end
      end

      dft_cube = FFTW.dft(cube)

      dft_cube.flatten.zip(dft_cube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitrary size of 3d inverse DFT" do
      cube = Array.new(8) { Array.new(8) { Array.new(8) { Complex.new(r.next_u, r.next_u) } } }

      cube_back = FFTW.idft(FFTW.dft(cube))

      cube_back.flatten.zip(cube.flatten) do |a, b|
        a.should be_close(b * 512.0, 1e-2)
      end
    end

    it "can compute abitrary size of high dimensional DFT" do
      hcube = Array.new(4) { Array.new(4) { Array.new(4) { Array.new(4) { Complex.new(r.next_u, r.next_u) } } } }
      dft_hcube_fallback = Array.new(4) do |i0|
        Array.new(4) do |i1|
          Array.new(4) do |i2|
            Array.new(4) do |i3|
              x = Complex.new(0, 0)

              hcube.each_with_index do |cube, j0|
                cube.each_with_index do |mat, j1|
                  mat.each_with_index do |ary, j2|
                    ary.each_with_index do |y, j3|
                      c = (i0 * j0 + i1 * j1 + i2 * j2 + i3 * j3) / 2.0
                      x += y * Complex.new(Math.cos(Math::PI * c), -Math.sin(Math::PI * c))
                    end
                  end
                end
              end

              x
            end
          end
        end
      end

      dft_hcube = FFTW.dft(hcube)

      dft_hcube.flatten.zip(dft_hcube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitrary size of high dimensional inverse DFT" do
      hcube = Array.new(4) { Array.new(4) { Array.new(4) { Array.new(4) { Complex.new(r.next_u, r.next_u) } } } }

      hcube_back = FFTW.idft(FFTW.dft(hcube))

      hcube_back.flatten.zip(hcube.flatten) do |a, b|
        a.should be_close(b * 256, 1e-2)
      end
    end

    it "can compute DFT with given dimensions" do
      mat = Array.new(32) { Array.new(32) { Complex.new(r.next_u, r.next_u) } }
      ary = mat.flatten

      dft_mat = FFTW.dft(mat)
      dft_ary = FFTW.dft(ary, 32, 32)

      dft_mat.flatten.zip(dft_ary) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT with given dimensions" do
      mat = Array.new(32) { Array.new(32) { Complex.new(r.next_u, r.next_u) } }
      ary = mat.flatten

      ary_back = FFTW.idft(FFTW.dft(ary, 32, 32), 32, 32)

      ary_back.zip(ary) do |a, b|
        a.should be_close(b * 1024, 1e-2)
      end
    end
  end

  describe "DFT of Real Data" do
    it "can compute abitrary size of 1d DFT" do
      arr = Array.new(32) { r.next_u.to_f }
      dft_arr_fallback = Array.new(17) do |i|
        x = Complex.new(0, 0)
        arr.each_with_index do |y, j|
          x += y * Complex.new(Math.cos(Math::PI * i * j / 16.0), -Math.sin(Math::PI * i * j / 16.0))
        end
        x
      end

      dft_arr = FFTW.dft(arr)

      dft_arr.zip(dft_arr_fallback) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitrary size of 1d inverse DFT" do
      arr = Array.new(32) { r.next_u.to_f }

      arr_back = FFTW.idft_r(FFTW.dft(arr), 32)

      arr_back.zip(arr) do |a, b|
        a.should be_close(b * 32, 1e-2)
      end
    end

    it "can compute abitrary size of 2d DFT" do
      mat = Array.new(8) { Array.new(8) { r.next_u.to_f } }
      dft_mat_fallback = Array.new(8) do |i|
        Array.new(5) do |j|
          x = Complex.new(0, 0)
          mat.each_with_index do |ary, k|
            ary.each_with_index do |y, l|
              c = i * k / 4.0 + j * l / 4.0
              x += y * Complex.new(Math.cos(Math::PI * c), -Math.sin(Math::PI * c))
            end
          end
          x
        end
      end

      dft_mat = FFTW.dft(mat)

      dft_mat.flatten.zip(dft_mat_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitrary size of 2d inverse DFT" do
      mat = Array.new(8) { Array.new(8) { r.next_u.to_f } }

      mat_back = FFTW.idft_r(FFTW.dft(mat), 8)

      mat_back.flatten.zip(mat.flatten) do |a, b|
        a.should be_close(b * 64, 1e-2)
      end
    end

    it "can compute abitrary size of 3d DFT" do
      cube = Array.new(8) { Array.new(8) { Array.new(8) { r.next_u.to_f } } }
      dft_cube_fallback = Array.new(8) do |i1|
        Array.new(8) do |i2|
          Array.new(5) do |i3|
            x = Complex.new(0, 0)

            cube.each_with_index do |mat, j1|
              mat.each_with_index do |ary, j2|
                ary.each_with_index do |y, j3|
                  c = (i1 * j1 + i2 * j2 + i3 * j3) / 4.0

                  x += y * Complex.new(Math.cos(Math::PI * c), -Math.sin(Math::PI * c))
                end
              end
            end
            x
          end
        end
      end

      dft_cube = FFTW.dft(cube)

      dft_cube.flatten.zip(dft_cube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute abitrary size of 3d inverse DFT" do
      cube = Array.new(8) { Array.new(8) { Array.new(8) { r.next_u.to_f } } }

      cube_back = FFTW.idft_r(FFTW.dft(cube), 8)

      cube_back.flatten.zip(cube.flatten) do |a, b|
        a.should be_close(b * 512, 1e-2)
      end
    end

    it "can compute arbitrary size of high dimensional DFT" do
      hcube = Array.new(4) { Array.new(4) { Array.new(4) { Array.new(4) { r.next_u.to_f } } } }
      dft_hcube_fallback = Array.new(4) do |i0|
        Array.new(4) do |i1|
          Array.new(4) do |i2|
            Array.new(3) do |i3|
              x = Complex.new(0, 0)

              hcube.each_with_index do |cube, j0|
                cube.each_with_index do |mat, j1|
                  mat.each_with_index do |ary, j2|
                    ary.each_with_index do |y, j3|
                      c = (i0 * j0 + i1 * j1 + i2 * j2 + i3 * j3) / 2.0
                      x += y * Complex.new(Math.cos(Math::PI * c), -Math.sin(Math::PI * c))
                    end
                  end
                end
              end

              x
            end
          end
        end
      end

      dft_hcube = FFTW.dft(hcube)

      dft_hcube.flatten.zip(dft_hcube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "compute abitrary size of high dimensional inverse DFT" do
      hcube = Array.new(4) { Array.new(4) { Array.new(4) { Array.new(4) { r.next_u.to_f } } } }

      hcube_back = FFTW.idft_r(FFTW.dft(hcube), 4)

      hcube_back.flatten.zip(hcube.flatten) do |a, b|
        a.should be_close(b * 256, 1e-2)
      end
    end

    it "can compute DFT with given dimensions" do
      mat = Array.new(32) { Array.new(32) { r.next_u.to_f } }
      ary = mat.flatten

      dft_mat = FFTW.dft(mat)
      dft_ary = FFTW.dft(ary, 32, 32)

      dft_mat.flatten.zip(dft_ary) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT with given dimensions" do
      mat = Array.new(32) { Array.new(32) { r.next_u.to_f } }
      ary = mat.flatten

      ary_back = FFTW.idft_r(FFTW.dft(ary, 32, 32), 32, 32)

      ary_back.zip(ary) do |a, b|
        a.should be_close(b * 1024, 1e-2)
      end
    end
  end
end
