require "./spec_helper"

r = Random.new

describe FFTW::Plan do
  describe "1-Dimension plan" do
    plan = FFTW::Plan.new(32)
    it "can compute DFT of complex data" do
      arr = Array.new(32) { Complex.new(r.next_u, r.next_u) }
      dft_arr_fallback = Array.new(32) do |i|
        x = Complex.new(0, 0)
        arr.each_with_index do |a, j|
          x += a * Complex.new(Math.cos(Math::PI * i * j / 16.0), -Math.sin(Math::PI * i * j / 16.0))
        end
        x
      end

      dft_arr = plan.dft(arr)

      dft_arr.zip(dft_arr_fallback) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of complex data" do
      arr = Array.new(32) { Complex.new(r.next_u, r.next_u) }

      arr_back = plan.idft(plan.dft(arr))

      arr_back.zip(arr) do |a, b|
        a.should be_close(b * 32, 1e-2)
      end
    end

    it "can compute DFT of real data" do
      arr = Array.new(32) { r.next_u.to_f }
      dft_arr_fallback = Array.new(17) do |i|
        x = Complex.new(0, 0)
        arr.each_with_index do |a, j|
          x += a * Complex.new(Math.cos(Math::PI * i * j / 16.0), -Math.sin(Math::PI * i * j / 16.0))
        end
        x
      end

      dft_arr = plan.dft(arr)

      dft_arr.zip(dft_arr_fallback) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of real data" do
      arr = Array.new(32) { r.next_u.to_f }

      arr_back = plan.idft_r(plan.dft(arr))

      arr_back.zip(arr) do |a, b|
        a.should be_close(b * 32, 1e-2)
      end
    end
  end

  describe "2-Dimension plan" do
    plan = FFTW::Plan.new(8, 8)
    it "can compute DFT of complex data" do
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

      dft_mat_flat = plan.dft(mat.flatten)

      dft_mat_flat.zip(dft_mat_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can computer inverse DFT of complex data" do
      mat = Array.new(8) { Array.new(8) { Complex.new(r.next_u, r.next_u) } }
      mat_flat = mat.flatten

      mat_flat_back = plan.idft(plan.dft(mat_flat))

      mat_flat_back.zip(mat_flat) do |a, b|
        a.should be_close(b * 64, 1e-2)
      end
    end

    it "can compute DFT of real data" do
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

      dft_mat_flat = plan.dft(mat.flatten)

      dft_mat_flat.zip(dft_mat_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of real data" do
      mat = Array.new(8) { Array.new(8) { r.next_u.to_f } }
      mat_flat = mat.flatten

      mat_flat_back = plan.idft_r(plan.dft(mat_flat))

      mat_flat_back.zip(mat_flat) do |a, b|
        a.should be_close(b * 64, 1e-2)
      end
    end
  end

  describe "3-Dimension plan" do
    plan = FFTW::Plan.new(8, 8, 8)
    it "can compute DFT of complex data" do
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

      dft_cube_flat = plan.dft(cube.flatten)

      dft_cube_flat.zip(dft_cube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of complex data" do
      cube = Array.new(8) { Array.new(8) { Array.new(8) { Complex.new(r.next_u, r.next_u) } } }
      cube_flat = cube.flatten

      cube_flat_back = plan.idft(plan.dft(cube_flat))

      # Weird result
      cube_flat.zip(cube_flat) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute DFT of real data" do
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

      dft_cube_flat = plan.dft(cube.flatten)

      dft_cube_flat.zip(dft_cube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of real data" do
      cube = Array.new(8) { Array.new(8) { Array.new(8) { r.next_u.to_f } } }
      cube_flat = cube.flatten

      cube_flat_back = plan.idft_r(plan.dft(cube_flat))

      # Weird result
      cube_flat.zip(cube_flat) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end
  end

  describe "High dimensional plan" do
    plan = FFTW::Plan.new(4, 4, 4, 4)
    it "can compute DFT of complex data" do
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

      dft_hcube_flat = plan.dft(hcube.flatten)

      dft_hcube_flat.zip(dft_hcube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of complex data" do
      hcube = Array.new(4) { Array.new(4) { Array.new(4) { Array.new(4) { Complex.new(r.next_u, r.next_u) } } } }
      hcube_flat = hcube.flatten

      hcube_back_flat = plan.idft(plan.dft(hcube_flat))

      hcube_back_flat.zip(hcube_flat) do |a, b|
        a.should be_close(b * 256, 1e-2)
      end
    end

    it "can compute DFT of real data" do
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

      dft_hcube_flat = plan.dft(hcube.flatten)

      dft_hcube_flat.zip(dft_hcube_fallback.flatten) do |a, b|
        a.should be_close(b, 1e-2)
      end
    end

    it "can compute inverse DFT of real data" do
      hcube = Array.new(4) { Array.new(4) { Array.new(4) { Array.new(4) { r.next_u.to_f } } } }
      hcube_flat = hcube.flatten

      hcube_back_flat = plan.idft_r(plan.dft(hcube_flat))

      hcube_back_flat.zip(hcube_flat) do |a, b|
        a.should be_close(b * 256, 1e-2)
      end
    end
  end
end
