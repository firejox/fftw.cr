# fftw.cr

[![Linux CI](https://github.com/firejox/fftw.cr/actions/workflows/linux.yml/badge.svg)](https://github.com/firejox/fftw.cr/actions/workflows/linux.yml)
[![MacOSX CI](https://github.com/firejox/fftw.cr/actions/workflows/macos.yml/badge.svg)](https://github.com/firejox/fftw.cr/actions/workflows/macos.yml)
[![Windows CI](https://github.com/firejox/fftw.cr/actions/workflows/windows.yml/badge.svg)](https://github.com/firejox/fftw.cr/actions/workflows/windows.yml) 

Crystal wrapper to the [FFTW 3](http://www.fftw.org) library

## Installation

First install fftw:

```bash
sudo pacman -S fftw
```

Add this to your application's `shard.yml`:

```yaml
dependencies:
  fftw.cr:
    github: firejox/fftw.cr
```

## Usage

You can compute abtitrary size of Fourier transform by this:

```crystal
require "fftw.cr"

x = Array.new(512) { Complex.new(Random.next_u, Random.next_u) }

dft_x = FFTW.dft(x)
```

Or be more efficient on fix-size of transform by this:

```crystal
require "fftw.cr"

plan = FFTW::Plan.new(512)

x = Array.new(512) { Complex.new(Random.next_u, Random.next_u) }

dft_x = plan.dft(x)
```

## Development

### Roadmap

- [x] Add Plan class
- [ ] Add full transformations listed in [FFTW3 doc](http://www.fftw.org/fftw3_doc/Tutorial.html#Tutorial)
- [ ] FFTW3 wisdom feature
- [ ] Multithread
- [ ] MPI support

## Contributing

1. Fork it ( https://github.com/firejox/fftw.cr/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [[firejox]](https://github.com/firejox) firejox - creator, maintainer
