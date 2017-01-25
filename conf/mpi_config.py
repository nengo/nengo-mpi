import sys, os, platform

from distutils.util  import split_quoted
from distutils.spawn import find_executable
from distutils import log as default_log
from distutils.errors import DistutilsPlatformError
from distutils import sysconfig

from textwrap import dedent

from collections import OrderedDict


# Best not to bother with six/future since the is called during setup.
try:
    from configparser import ConfigParser
    from configparser import Error as ConfigParserError
except ImportError:
    from ConfigParser import ConfigParser
    from ConfigParser import Error as ConfigParserError


def fix_compiler_cmd(cc, mpicc):
    if not mpicc: return
    i = 0
    while os.path.basename(cc[i]) == 'env':
        i = i + 1
        while '=' in cc[i]:
            i = i + 1
    while os.path.basename(cc[i]) == 'ccache':
        i = i + 1
    cc[i:i+1] = split_quoted(mpicc)


def fix_linker_cmd(ld, mpild):
    if not mpild: return
    i = 0
    if (sys.platform.startswith('aix') and
        os.path.basename(ld[i]) == 'ld_so_aix'):
        i = 1
    while os.path.basename(ld[i]) == 'env':
        i = i + 1
        while '=' in ld[i]:
            i = i + 1
    while os.path.basename(ld[i]) == 'ccache':
        del ld[i]
    ld[i:i+1] = split_quoted(mpild)



def get_configuration(command_obj, verbose=True):
    config = Config()
    config.setup(command_obj)
    if verbose:
        if config.section and config.filename:
            default_log.info("MPI configuration: [%s] from '%s'",
                     config.section, ','.join(config.filename))
            config.info(default_log)
    return config


def configure_compiler(compiler, config, lang=None):
    """ Uses config to customize compiler, and sets certain other flags. """

    mpicc = config.get('mpicc')
    mpicxx = config.get('mpicxx')
    mpild = config.get('mpild')
    if not mpild and (mpicc or mpicxx):
        if lang == 'c': mpild = mpicc
        if lang == 'c++': mpild = mpicxx
        if not mpild: mpild = mpicc or mpicxx

    customize_compiler(compiler, lang,
                       mpicc=mpicc, mpicxx=mpicxx, mpild=mpild)

    for k, v in config.get('define_macros', []):
        compiler.define_macro(k, v)
    for v in config.get('undef_macros', []):
        compiler.undefine_macro(v)
    for v in config.get('include_dirs', []):
        compiler.add_include_dir(v)
    for v in config.get('libraries', []):
        compiler.add_library(v)
    for v in config.get('library_dirs', []):
        compiler.add_library_dir(v)
    for v in config.get('runtime_library_dirs', []):
        compiler.add_runtime_library_dir(v)
    for v in config.get('extra_objects', []):
        compiler.add_link_object(v)

    cc_args = config.get('extra_compile_args', [])
    ld_args = config.get('extra_link_args', [])
    compiler.compiler += cc_args
    compiler.compiler_so += cc_args
    compiler.compiler_cxx += cc_args
    compiler.linker_so += ld_args
    compiler.linker_exe += ld_args

    return compiler


def customize_compiler(compiler, lang=None, mpicc=None, mpicxx=None, mpild=None):
    """ Implements the compiler configuration. """

    # Unix specific compilation customization
    assert compiler.compiler_type == 'unix'

    sysconfig.customize_compiler(compiler)

    ld = compiler.linker_exe
    for envvar in ('LDFLAGS', 'CFLAGS', 'CPPFLAGS'):
        if envvar in os.environ:
            ld += split_quoted(os.environ[envvar])

    # Compiler command overriding
    if mpicc:
        fix_compiler_cmd(compiler.compiler, mpicc)
        if lang in ('c', None):
            fix_compiler_cmd(compiler.compiler_so, mpicc)

    if mpicxx:
        fix_compiler_cmd(compiler.compiler_cxx, mpicxx)
        if lang == 'c++':
            fix_compiler_cmd(compiler.compiler_so, mpicxx)

    if mpild:
        for ld in [compiler.linker_so, compiler.linker_exe]:
            fix_linker_cmd(ld, mpild)

    badcxxflags = ['-Wimplicit', '-Wstrict-prototypes']
    for flag in badcxxflags:
        while flag in compiler.compiler_cxx:
            compiler.compiler_cxx.remove(flag)
        if lang == 'c++':
            while flag in compiler.compiler_so:
                compiler.compiler_so.remove(flag)


def configure_mpi_sim(ext, config_cmd):
    headers = ['stdlib.h', 'mpi.h']

    default_log.info("checking for MPI compile and link ...")
    ConfigTest = dedent("""\
    int main(int argc, char **argv)
    {
      (void)MPI_Init(&argc, &argv);
      (void)MPI_Finalize();
      return 0;
    }
    """)

    errmsg = "Cannot %s MPI programs. Check your configuration!!!"

    ok = config_cmd.try_compile(ConfigTest, headers=headers)
    if not ok:
        raise DistutilsPlatformError(errmsg % "compile")

    ok = config_cmd.try_link(ConfigTest, headers=headers)
    if not ok:
        raise DistutilsPlatformError(errmsg % "link")

    default_log.info("checking for missing MPI functions/symbols ...")
    tests = ["defined(%s)" % macro for macro in
             ("OPEN_MPI", "MPICH2", "DEINO_MPI", "MSMPI_VER",)]
    tests += ["(defined(MPICH_NAME)&&(MPICH_NAME==3))"]
    ConfigTest = dedent("""\
    #if !(%s)
    #error "Unknown MPI implementation"
    #endif
    """) % "||".join(tests)
    ok = config_cmd.try_compile(ConfigTest, headers=headers)
    if not ok:
        from mpidistutils import ConfigureMPI
        configure = ConfigureMPI(config_cmd)
        results = configure.run()
        configure.dump(results)
        ext.define_macros += [('HAVE_CONFIG_H', 1)]

    if os.name == 'posix':
        configure_dl(ext, config_cmd)


# A minimal MPI program

ConfigTest = """\
int main(int argc, char **argv)
{
  int ierr;
  (void)argc; (void)argv;
  ierr = MPI_Init(&argc, &argv);
  if (ierr) return -1;
  ierr = MPI_Finalize();
  if (ierr) return -1;
  return 0;
}
"""


class Config(object):

    def __init__(self, log=None):
        self.log = log or default_log
        self.section  = None
        self.filename = None
        self.compiler_info = OrderedDict((
                ('mpicc'  , None),
                ('mpicxx' , None),
                ('mpifort', None),
                ('mpif77' , None),
                ('mpif90' , None),
                ('mpif08' , None),
                ('mpild'  , None),
                ))
        self.library_info = OrderedDict((
            ('define_macros'        , []),
            ('undef_macros'         , []),
            ('include_dirs'         , []),

            ('libraries'            , []),
            ('library_dirs'         , []),
            ('runtime_library_dirs' , []),

            ('extra_compile_args'   , []),
            ('extra_link_args'      , []),
            ('extra_objects'        , []),
            ))

    def __bool__(self):
        for v in self.compiler_info.values():
            if v:
                return True
        for v in self.library_info.values():
            if v:
                return True
        return False

    __nonzero__ = __bool__

    def get(self, k, d=None):
        if k in self.compiler_info:
            return self.compiler_info[k]
        if k in self.library_info:
            return self.library_info[k]
        return d

    def info(self, log=None):
        log = log or self.log

        mpicc   = self.compiler_info.get('mpicc')
        mpicxx  = self.compiler_info.get('mpicxx')
        mpifort = self.compiler_info.get('mpifort')
        mpif77  = self.compiler_info.get('mpif77')
        mpif90  = self.compiler_info.get('mpif90')
        mpif08  = self.compiler_info.get('mpif08')
        mpild   = self.compiler_info.get('mpild')
        if mpicc:
            log.info("MPI C compiler:    %s", mpicc)
        if mpicxx:
            log.info("MPI C++ compiler:  %s", mpicxx)
        if mpifort:
            log.info("MPI F compiler:    %s", mpifort)
        if mpif77:
            log.info("MPI F77 compiler:  %s", mpif77)
        if mpif90:
            log.info("MPI F90 compiler:  %s", mpif90)
        if mpif08:
            log.info("MPI F08 compiler:  %s", mpif08)
        if mpild:
            log.info("MPI linker:        %s", mpild)

    def update(self, config, **more):
        if hasattr(config, 'keys'):
            config = config.items()
        for option, value in config:
            if option in self.compiler_info:
                self.compiler_info[option] = value
            if option in self.library_info:
                self.library_info[option] = value
        if more:
            self.update(more)

    def setup(self, options, environ=None):
        if environ is None: environ = os.environ
        self.setup_library_info(options, environ)
        self.setup_compiler_info(options, environ)

    def setup_library_info(self, options, environ):
        filename = section = None
        mpiopt = getattr(options, 'mpi', None)
        mpiopt = environ.get('MPICFG', mpiopt)
        if mpiopt:
            if ',' in mpiopt:
                section, filename = mpiopt.split(',', 1)
            elif os.path.isfile(mpiopt):
                filename = mpiopt
            else:
                section = mpiopt
        if not filename: filename = "mpi.cfg"
        if not section:  section  = "mpi"

        mach = platform.machine()
        arch = platform.architecture()[0]
        plat = sys.platform
        osnm = os.name
        if   'linux' == plat[:5]: plat = 'linux'
        elif 'sunos' == plat[:5]: plat = 'solaris'
        elif 'win'   == plat[:3]: plat = 'windows'
        suffixes = []
        suffixes.append(plat+'-'+mach)
        suffixes.append(plat+'-'+arch)
        suffixes.append(plat)
        suffixes.append(osnm+'-'+mach)
        suffixes.append(osnm+'-'+arch)
        suffixes.append(osnm)
        suffixes.append(mach)
        suffixes.append(arch)
        sections  = [section+"-"+s for s in suffixes]
        sections += [section]
        self.load(filename, sections)

        if not self:
            raise Exception("Failed loading configuration.")

    def setup_compiler_info(self, options, environ):
        def find_exe(cmd, path=None):
            if not cmd: return None
            parts = split_quoted(cmd)
            exe, args = parts[0], parts[1:]
            if not os.path.isabs(exe) and path:
                exe = os.path.basename(exe)
            exe = find_executable(exe, path)
            if not exe: return None
            return ' '.join([exe]+args)
        COMPILERS = (
            ('mpicc',   ['mpicc',   'mpcc_r']),
            ('mpicxx',  ['mpicxx',  'mpic++', 'mpiCC', 'mpCC_r']),
            ('mpifort', ['mpifort', 'mpfort_r']),
            ('mpif77',  ['mpif77',  'mpf77_r']),
            ('mpif90',  ['mpif90',  'mpf90_r']),
            ('mpif08',  ['mpif08',  'mpf08_r']),
            ('mpild',   []),
            )

        compiler_info = {}
        PATH = environ.get('PATH', '')
        for name, _ in COMPILERS:
            cmd = (environ.get(name.upper()) or
                   getattr(options, name, None) or
                   self.compiler_info.get(name) or
                   None)
            if cmd:
                exe = find_exe(cmd, path=PATH)
                if exe:
                    path = os.path.dirname(exe)
                    PATH = path + os.path.pathsep + PATH
                    compiler_info[name] = exe
                else:
                    self.log.error("error: '%s' not found", cmd)

        if not self and not compiler_info:
            for name, candidates in COMPILERS:
                for cmd in candidates:
                    cmd = find_exe(cmd)
                    if cmd:
                        compiler_info[name] = cmd
                        break

        self.compiler_info.update(compiler_info)


    def load(self, filename="mpi.cfg", section='mpi'):
        if isinstance(filename, str):
            filenames = filename.split(os.path.pathsep)
        else:
            filenames = list(filename)
        if isinstance(section, str):
            sections = section.split(',')
        else:
            sections = list(section)

        try:
            parser = ConfigParser(dict_type=OrderedDict)
        except TypeError:
            parser = ConfigParser()
        try:
            read_ok = parser.read(filenames)
        except ConfigParserError:
            self.log.error(
                "error: parsing configuration file/s '%s'",
                os.path.pathsep.join(filenames))
            return None
        for section in sections:
            if parser.has_section(section):
                break
            section = None
        if not section:
            self.log.error(
                "error: section/s '%s' not found in file/s '%s'",
                ','.join(sections), os.path.pathsep.join(filenames))
            return None
        parser_items = list(parser.items(section, vars=None))

        compiler_info = type(self.compiler_info)()
        for option, value in parser_items:
            if option in self.compiler_info:
                compiler_info[option] = value

        pathsep = os.path.pathsep
        expanduser = os.path.expanduser
        expandvars = os.path.expandvars
        library_info = type(self.library_info)()
        for k, v in parser_items:
            if k in ('define_macros',
                     'undef_macros',
                     ):
                macros = [e.strip() for e in v.split(',')]
                if k == 'define_macros':
                    for i, m in enumerate(macros):
                        try: # -DFOO=bar
                            idx = m.index('=')
                            macro = (m[:idx], m[idx+1:] or None)
                        except ValueError: # -DFOO
                            macro = (m, None)
                        macros[i] = macro
                library_info[k] = macros
            elif k in ('include_dirs',
                       'library_dirs',
                       'runtime_dirs',
                       'runtime_library_dirs',
                       ):
                if k == 'runtime_dirs': k = 'runtime_library_dirs'
                pathlist = [p.strip() for p in v.split(pathsep)]
                library_info[k] = [expanduser(expandvars(p))
                                   for p in pathlist if p]
            elif k == 'libraries':
                library_info[k] = [e.strip() for e in split_quoted(v)]
            elif k in ('extra_compile_args',
                       'extra_link_args',
                       ):
                library_info[k] = split_quoted(v)
            elif k == 'extra_objects':
                library_info[k] = [expanduser(expandvars(e))
                                   for e in split_quoted(v)]
            elif hasattr(self, k):
                library_info[k] = v.strip()
            else:
                pass

        self.section = section
        self.filename = read_ok
        self.compiler_info.update(compiler_info)
        self.library_info.update(library_info)
        return compiler_info, library_info, section, read_ok

    def dump(self, filename=None, section='mpi'):
        # prepare configuration values
        compiler_info   = self.compiler_info.copy()
        library_info = self.library_info.copy()
        for k in library_info:
            if k in ('define_macros',
                     'undef_macros',
                     ):
                macros = library_info[k]
                if k == 'define_macros':
                    for i, (m, v) in enumerate(macros):
                        if v is None:
                            macros[i] = m
                        else:
                            macros[i] = '%s=%s' % (m, v)
                library_info[k] = ','.join(macros)
            elif k in ('include_dirs',
                       'library_dirs',
                       'runtime_library_dirs',
                       ):
                library_info[k] = os.path.pathsep.join(library_info[k])
            elif isinstance(library_info[k], list):
                library_info[k] = ' '.join(library_info[k])
        # fill configuration parser
        try:
            parser = ConfigParser(dict_type=OrderedDict)
        except TypeError:
            parser = ConfigParser()
        parser.add_section(section)
        for option, value in compiler_info.items():
            if not value: continue
            parser.set(section, option, value)
        for option, value in library_info.items():
            if not value: continue
            parser.set(section, option, value)
        # save configuration file
        if filename is None:
            parser.write(sys.stdout)
        elif hasattr(filename, 'write'):
            parser.write(filename)
        elif isinstance(filename, str):
            with open(filename, 'w') as f:
                parser.write(f)
        return parser


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("--mpi",     type="string")
    parser.add_option("--mpicc",   type="string")
    parser.add_option("--mpicxx",  type="string")
    parser.add_option("--mpifort", type="string")
    parser.add_option("--mpif77",  type="string")
    parser.add_option("--mpif90",  type="string")
    parser.add_option("--mpif08",  type="string")
    parser.add_option("--mpild",   type="string")
    (opts, args) = parser.parse_args()

    cfg = Config(log.Log(log))
    cfg.setup(opts)
    cfg.dump()
