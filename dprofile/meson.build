# >>> project definition
project('DProfile', ['CPP'],
	# Get version number from file.
	version: run_command(find_program('cat'),files('./VERSION'), check: true).stdout().strip(),
)
project_name = 'dprofile'
project_name_abbreviation = 'dp'

add_project_arguments('-O0',  language : 'cpp')

# >>>>>>>>>>>>>> build prepare >>>>>>>>>>>>>>
# get environment variables of the build process
env = environment()

# we use pkg-config to collect dependecies
pkgconfig = find_program('pkg-config')
bash = find_program('bash')

# compile flags
c_args = []

# load flags
ld_args = []

# include directories
inc_dirs = []

# source files
sources = []

# current directory of this meson file
root_dir = meson.current_source_dir()

# file for scanning source
scan_src_path = root_dir + '/scripts/glob_src.py'


# extract cuda version
cuda_version = run_command(bash, './scripts/get_cuda_version.sh').stdout().strip()
cuda_pc_path = '/usr/lib/pkgconfig'
cuda_modules = ['cublas', 'cuda', 'cudart', 'cufft', 'cufftw', 'cuinj64', 'curand', 'cusolver', 'cusparse']
founded_cuda_modules = []
message('>>> Detecting CUDA toolkit, assume:')
message('>>>>>> CUDA version:               ' + cuda_version)
message('>>>>>> Path of pkg-config files:   ' + cuda_pc_path)

env.set('PKG_CONFIG_PATH', cuda_pc_path)
foreach cuda_module : cuda_modules
    cuda_module_cflags = run_command(pkgconfig, '--cflags', cuda_module+'-'+cuda_version, env: env, check: false)
    cuda_module_ldflags = run_command(pkgconfig, '--libs', '--static', cuda_module+'-'+cuda_version, env: env, check: false) # add '--static' option if static link is needed
    cuda_module_version = run_command(pkgconfig, '--modversion', cuda_module+'-'+cuda_version, env: env, check: false)
    if cuda_module_cflags.returncode() != 0 or cuda_module_ldflags.returncode() != 0 or cuda_module_version.returncode() != 0
        message('>>>>>> Failed to find ' + cuda_module+'-'+cuda_version)
    else
        founded_cuda_modules += cuda_module
        c_args += cuda_module_cflags.stdout().split()
        ld_args += cuda_module_ldflags.stdout().split()
        message('>>>>>> Found ' + cuda_module + ', version is ' + cuda_module_version.stdout().split()[0])
    endif
endforeach
assert(
    founded_cuda_modules.length() == cuda_modules.length(),
    'Only find ' + founded_cuda_modules.length().to_string() + ' CUDA modules, expected ' + cuda_modules.length().to_string()
)
message('>>> Detecting CUDA toolkit done\n')


# >>>>>>>>>>>>>> options >>>>>>>>>>>>>>
enable_print_error = 1
enable_print_warn = 1
enable_print_log = 1
enable_print_debug = 1
enable_print_with_color = 1

# >>>>>>>>>>>>>> source and includes >>>>>>>>>>>>>>
sources += run_command('python3', files(scan_src_path), './src', check: false).stdout().strip().split('\n')
inc_dirs += [ './' ]

# add CUPTI and NVPerf, it's under /usr/local/cuda/lib64/libcupti.so libnvperf_host.so libnvperf_target.so
ld_args += [ '-lcupti', '-lnvperf_host', '-lnvperf_target', '-lnvidia-ml' ]

# add gpu metrics
inc_dirs += [ './gpu-metrics' ]

# add pybind11
inc_dirs += [ './third_party/pybind11/include' ]
ld_args += ['-L' + root_dir + '/third_party/pybind11/build/tests']

# add python3-dev
# python3-config --ldflags
ld_args += [ '-L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu', '-L/usr/lib/x86_64-linux-gnu', '-lpython3.12', '-lcrypt', '-ldl', '-lm' ]
# python3-config --cflags 
c_args += [
    '-I/usr/include/python3.12',
    '-Wno-unused-result',
    '-Wsign-compare',
    '-g',
    '-fstack-protector-strong',
    '-Wformat',
    '-Werror=format-security',
    '-DNDEBU',
    '-fwrapv',
    '-O2',
    '-Wall'
]

# add libeigen
inc_dirs += [ '/usr/include/eigen3' ]

# >>>>>>>>>>>>>> setup build options >>>>>>>>>>>>>>
c_args += ['--std=c++17']

subdir('./utils')

library(
    project_name_abbreviation,
    sources,
    cpp_args: c_args,
	link_args: ld_args,
	include_directories: inc_dirs,
	install: false
)
