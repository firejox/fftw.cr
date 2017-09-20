 require "../src/*"

r = Random.new

c_arr = Array.new(512) { Complex.new(r.next_u, r.next_u) }

puts "Complex Array:"
c_arr.each do |x|
  puts x
end

c_arr_out = FFTW.dft c_arr

puts "The Discrete Fourier Transform of Complex Array:"
c_arr_out.each do |x|
  puts x
end

c_arr_back = FFTW.idft c_arr_out

puts "The Inverse Transform of Discrete Fourier Transform Complex Array:"
c_arr_back.each do |x|
  puts x
end


r_arr = Array.new(512) { r.next_u.to_f64 }

puts "Real Array:"
r_arr.each do |x|
  puts x
end

r_arr_out = FFTW.dft r_arr

puts "The Discrete Fourier Transform of Real Array:"
r_arr_out.each do |x|
  puts x
end

r_arr_back = FFTW.idft_r r_arr_out

puts "The Inverse Transform of Discrete Fourier Transform Real Array:"
r_arr_back.each do |x|
  puts x
end

plan = FFTW::Plan.new(512)

c_arr_out_1 = plan.dft c_arr

puts "The Discrete Fourier Transform of Complex Array by Plan:"
c_arr_out_1.each do |x|
  puts x
end

c_arr_back_1 = plan.idft c_arr_out_1
puts "The Inverse Transform of Discrete Fourier Transform Complex Array by Plan:"
c_arr_back_1.each do |x|
  puts x
end

r_arr_out_1 = plan.dft r_arr

puts "The Discrete Fourier Transform of Real Array by Plan:"
r_arr_out_1.each do |x|
  puts x
end

r_arr_back_1 = plan.idft_r r_arr_out_1

puts "The Inverse Transform of Discrete Fourier Transform Real Array by Plan:"
r_arr_back_1.each do |x|
  puts x
end

