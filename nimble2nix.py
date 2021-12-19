#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p git nim nix-prefetch-git python3Packages.packaging
import argparse
import contextlib
from functools import cache
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
from urllib.request import urlopen

from packaging.version import parse as parse_version


def run(cmd, *args, check=True, **kwargs):
    logging.getLogger('run').debug('$ %s', ' '.join(map(shlex.quote, cmd)))
    return subprocess.run(cmd, *args, check=check, **kwargs)


def parse_nimble(nimble):
    # nimble seems to write warnings to stdout instead of stderr, so we use --silent
    p = run(['nimble', '--silent', 'dump', '--json', nimble],
            stdout=subprocess.PIPE, encoding='utf-8')
    return json.loads(p.stdout)


def find_nimble(dir):
    nimbles = [x for x in os.listdir(dir) if x.endswith('.nimble')]
    assert len(nimbles) == 1
    return os.path.join(dir, nimbles[0])


class GitCache(object):
    """ Temporary directory helper that runs git clone """

    logger = logging.getLogger('GitCache')

    def __init__(self):
        self._temp = None
        self._root = None
        self._paths = {}
        self.cloned = 0
        self.reused = 0

    def __enter__(self):
        self._temp = tempfile.TemporaryDirectory(prefix='nimble2nix.')
        self._root = self._temp.__enter__()
        self.logger.debug("cloning git repos to %s", self._root)
        return self

    def __exit__(self, type, value, traceback):
        self._temp.__exit__(type, value, traceback)
        self._temp = None
        self._root = None
        self.logger.debug("cloned %d repositories, avoided %d redundant clones",
                          self.cloned, self.reused)

    def get_path(self, name, url):
        counter = 1
        name = name.replace(os.sep, '_')
        while True:
            suffix = '' if counter == 1 else '-' + str(counter)
            path = os.path.join(self._root, name + suffix)
            if not os.path.exists(path):
                self._paths[url] = path
                return path
            counter += 1

    def clone(self, url, name):
        try:
            path = self._paths[url]
            self.logger.debug('reusing %r for %r', path, url)
            self.reused += 1
            return path
        except KeyError:
            pass
        path = self.get_path(name, url)
        run(['git', 'clone', '--', url, path])
        self.cloned += 1
        return path


class Packages(object):
    def __init__(self, name=None):
        if name is None:
            logging.info("downloading packages.json...")
            with urlopen('https://github.com/nim-lang/packages/raw/master/packages.json') as resp:
                self.packages = json.loads(resp.read().decode('utf-8'))
        else:
            logging.info("using %s...", name)
            with open(name, 'r', encoding='utf-8') as fp:
                self.packages = json.load(fp)

    @cache
    def get(self, name):
        for pkg in self.packages:
            if pkg['name'] == name:
                return pkg
        return {
            'url': name,
            'method': 'git',
        }


def check_version_range(version_range, version):
    kind = version_range['kind']
    if kind == 'verAny':
        return True
    elif kind == 'verIntersect':
        return check_version_range(version_range['verILeft'], version) and check_version_range(version_range['verIRight'], version)
    else:
        try:
            ver = parse_version(version_range['ver'])
            return {
                'verLater':     version >  ver,
                'verEqLater':   version >= ver,
                'verEarlier':   version <  ver,
                'verEqEarlier': version <= ver,
                'verEq':        version == ver,
            }[kind]
        except KeyError:
            logging.error("version range %r not supported", version_range)
            raise


def intersect_version_range(a, b):
    # TODO apply some logic
    return {
        'kind':      'verIntersect',
        'verILeft':  a,
        'verIRight': b,
    }


def format_version_range(version_range):
    kind = version_range['kind']
    if kind == 'verAny':
        return '*'
    elif kind == 'verIntersect':
        return '%s %s' % (format_version_range(version_range['verILeft']),
                          format_version_range(version_range['verIRight']))
    elif kind == 'verSpecial':
        return version_range['spe']
    else:
        return {
            'verLater':     '>',
            'verEqLater':   '>=',
            'verEarlier':   '<',
            'verEqEarlier': '<=',
            'verTilde':     '~=',
            'verCaret':     '^=',
            'verEq':        '',
        }[kind] + version_range['ver']


class Requirement(object):
    skip = {'nim'}  # FIXME respect the nim version requirements

    @classmethod
    def from_nimble_file(cls, nimble_file, packages, git_cache):
        reqs = []
        for req in parse_nimble(nimble_file)['requires']:
            if req['name'] not in cls.skip:
                reqs.append(cls(req, packages, git_cache))
        return reqs

    def __init__(self, req, packages, git_cache):
        self.name = req['name']
        self.version = req['ver']
        self._packages = packages
        self._git_cache = git_cache

    @property
    @cache
    def pkg(self):
        return self._packages.get(self.name)

    def find_latest_rev(self):
        assert self.pkg['method'] == 'git', "%r not supported, currently the only supported method is 'git'" % self.pkg['method']

        git_dir = self._git_cache.clone(self.pkg['url'], self.name)

        rev = None
        add_tag = False

        kind = self.version['kind']
        if kind == 'verSpecial':
            assert self.version['spe'].startswith('#')
            rev = self.version['spe'][1:]
            # TODO what about nim's `head`

            # keep the original to rev from the nimble file so we can re-add it
            # to the git repo later
            add_tag = True
        else:
            # get latest tag that satisfies the version range
            tags = run(['git', '-C', git_dir, 'tag', '--list'],
                       stdout=subprocess.PIPE, encoding='utf-8')
            tags = tags.stdout.split()
            for tag in tags:
                parsed_tag = parse_version(tag)
                if check_version_range(self.version, parsed_tag):
                    if rev is None or parsed_tag > parse_version(rev):
                        rev = tag

            if rev is None:
                # see if nimble file in HEAD has a required version
                logging.warning("%s: %s does not provide any tags, so we check if HEAD satisfies the version",
                                self.name, self.pkg['url'])
                info = parse_nimble(find_nimble(git_dir))
                if check_version_range(self.version, parse_version(info['version'])):
                    rev = 'HEAD'

        if rev is None:
            raise RuntimeError("%s: cannot satisfy %r" % (self.name, format_version_range(self.version)))

        # nix-prefetch-git does not work with remote branches and such, so we
        # convert rev to a commit hash
        try:
            commit_hash = run(['git', '-C', git_dir, 'rev-parse', rev],
                              stdout=subprocess.PIPE, encoding='utf-8').stdout.strip()
        except subprocess.CalledProcessError:
            # try again with remote branches
            commit_hash = run(['git', '-C', git_dir, 'rev-parse', 'remotes/origin/' + rev],
                              stdout=subprocess.PIPE, encoding='utf-8').stdout.strip()
        logging.info("%s: %s%s", self.name, commit_hash,
                     ' (%s)' % rev if rev != commit_hash else '')

        # do not add rev from nimble file to the git repo because it is
        # unimportant or an abbreviated commit hash
        if not add_tag or commit_hash.startswith(rev):
            rev = None

        return (commit_hash, rev)

    def prefetch(self):
        commit_hash, rev = self.find_latest_rev()

        # re-add rev from the nimble file to the git repository,
        # nix-prefetch-git removes almost everything and otherwise nimble will
        # not find the commit
        add_tag = '' if rev is None else \
            'git -C "$dir" tag -f %s %s >&2' % (shlex.quote(rev), shlex.quote(commit_hash))
        env = dict(os.environ)
        env['NIX_PREFETCH_GIT_CHECKOUT_HOOK'] = add_tag

        p = run(['nix-prefetch-git',
                 '--fetch-submodules',
                 '--leave-dotGit',
                 '--rev', commit_hash,
                 '--url', self.pkg['url']],
                env=env, stdout=subprocess.PIPE, encoding='utf-8')
        info = json.loads(p.stdout)
        info['NIX_PREFETCH_GIT_CHECKOUT_HOOK'] = add_tag
        return info


def dot_quote(x):
    return x.replace('\\', '\\\\').replace('"', '\\"').join('""')


def collect_requirements(nimble, write_dot, url=None, *, collected=None, **kwargs):
    # we will index requirements by their URL, whenever a requirement is
    # encountered store it, if it is already known we update its version range
    # and re-run the process on it
    # TODO thinking about it, this might add sub-requirements that a no longer
    # needed because their parent-dependencies are of another version

    if collected is None:
        collected = {}
    if url is None:
        url = 'file://' + urllib.parse.quote(nimble)

    for req in Requirement.from_nimble_file(nimble, **kwargs):
        write_dot('\t%s -> %s [label=%s];\n' % (dot_quote(url),
                                                dot_quote(req.pkg['url']),
                                                dot_quote(format_version_range(req.version))))

        inserted = collected.setdefault(req.pkg['url'], req)
        if inserted.version != req.version:
            # package URL is already known, update the version range and re-run
            inserted.version = intersect_version_range(inserted.version, req.version)
            logging.info("common requirement %s with %r",
                         req.pkg['url'], format_version_range(inserted.version))
        del req

        inserted.prefetched = inserted.prefetch()

        collect_requirements(find_nimble(inserted.prefetched['path']),
                             url=inserted.pkg['url'], write_dot=write_dot,
                             collected=collected, **kwargs)

    return collected


def nix_dump(x):
    if isinstance(x, (bool, int, float, str)):
        return json.dumps(x)
    elif isinstance(x, list):
        return ' '.join(itertools.chain('[', map(nix_dump, x), ']'))
    else:
        raise TypeError('cannot convert %r to a nix value' % x)


def to_nimble_nix(requirements, fp):
    logging.info("creating nimble.nix...")
    fp.write('''\
{ fetchgit, writeText }:
let
  packages = [
''')
    for req in requirements.values():
        pkg = {'tags': []}
        pkg.update(req.pkg)
        pkg['name'] = req.name
        if 'license' not in pkg or 'description' not in pkg:
            info = parse_nimble(find_nimble(req.prefetched['path']))
            pkg.setdefault('license', info['license'])
            pkg.setdefault('description', info['desc'])

        fp.write('    {\n')
        for k, v in pkg.items():
            fp.write('      %s = ' % k)
            if k == 'url':
                fp.write('''"file://" + (fetchgit {
        url = %s;
        rev = %s;
        sha256 = %s;
        fetchSubmodules = true;
        leaveDotGit = true;
      })''' % (nix_dump(req.prefetched['url']),
               nix_dump(req.prefetched['rev']),
               nix_dump(req.prefetched['sha256'])))

                if req.prefetched['NIX_PREFETCH_GIT_CHECKOUT_HOOK']:
                    fp.write('''.overrideAttrs ({ ... }: {
        # re-add rev from the nimble file to the git repository,
        # nix-prefetch-git removes almost everything and otherwise nimble will
        # not find the commit
        NIX_PREFETCH_GIT_CHECKOUT_HOOK = %s;
      })''' % nix_dump(req.prefetched['NIX_PREFETCH_GIT_CHECKOUT_HOOK']))

            else:
                fp.write(nix_dump(v))
            fp.write(';\n')
        fp.write('    }\n')

    fp.write('''  ];
in
  writeText "packages.json" (builtins.toJSON packages)
''')


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect nimble requirements",
        epilog="This tool creates a nix derivation that creates a nimble "
            "package.json. The created package.json includes the requirements "
            "of the given nimble files recursively with their `url` pointing "
            "to the nix store. By creating a symlink from "
            "$nimbleDir/packages_global.json to the created package.json "
            "nimble can fetch the requirements when sandboxed. Because only "
            "one version of a requirement is supported this may not always be "
            "able to resolve the dependencies.")
    p.add_argument('-o', '--output',
                   required=True,
                   help="Nix derivation that creates the package.json")
    p.add_argument('-P', '--packages',
                   help="use custom packages.json instead of downloading")
    p.add_argument('--dot',
                   help="output DOT graph of the requirements")
    p.add_argument('-v', '--verbose',
                   action='store_const',
                   default=logging.INFO,
                   const=logging.DEBUG,
                   help="verbose logging")
    p.add_argument('nimble_file', nargs='+')
    args = p.parse_args()

    logging.basicConfig(format=('\x1b[32m%s\x1b[39m' if sys.stderr.isatty() else '%s')
                            % '[%(asctime)s] %(levelname)-8s %(name)-8s %(message)s',
                        stream=sys.stderr, level=args.verbose)

    with contextlib.ExitStack() as stack:
        packages = Packages(args.packages)
        git_cache = stack.enter_context(GitCache())

        # write DOT graph
        if args.dot is None:
            write_dot = lambda x: None
        else:
            fp = stack.enter_context(open(args.dot, 'w', encoding='utf-8', buffering=1))
            fp.write('''digraph {
	node [fontname=monospace];
	edge [fontname=monospace];
''')
            stack.callback(fp.write, '}\n')
            write_dot = fp.write
            logging.debug("writing dependency graph to %r...", args.dot)

        collected = {}
        for nimble in args.nimble_file:
            collect_requirements(nimble, write_dot, collected=collected,
                                 packages=packages, git_cache=git_cache)

        with open(args.output, 'w', encoding='utf-8') as fp:
            to_nimble_nix(collected, fp)


if __name__ == '__main__':
    main()
