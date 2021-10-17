import glob
from re import compile as rcompile


def get_docs(pyfile: str):
    docs = []
    regex = rcompile(
        r'^\s*(?:def|class)\s([\w_]+\(.*\)?[^:\n]):?$|^\s*(?!(?:if|def|for|while|elif|else|class)\s)([\w()=_,*\s\\\'\"]+\)):$|.*#\s*(.+)$')
    i = 0
    with open(pyfile, 'rt') as file:
        lines = file.readlines()
        while i < len(lines):
            if 'def ' in lines[i] or 'class ' in lines[i] and not lines[i].strip()[0] == '#':
                # matches = regex.match(lines[i])
                # if matches is None:
                #     print()
                grp1, _, grp3 = regex.match(lines[i]).groups()
                assert _ is None, 'Group 2 found when matching def/class.'
                grp1 = grp1.strip(' \n')
                line = grp1
                comm = grp3 or ''
                while '):' not in lines[i]:
                    i += 1
                    if '):' not in lines[i]:
                        line += lines[i].strip(' \n')
                    else:
                        # matches = regex.match(lines[i])
                        # if matches is None:
                        #     print()
                        _, grp2, grp3 = regex.match(lines[i]).groups()
                        grp2 = grp2 or ''
                        assert _ is None, 'Group 1 found when matching for end of def/class line.'
                        comm += ('\n' + grp3) if grp3 is not None else ''
                        line += grp2
                docs.append(line + '\n')
                docs.append(comm + '\n')
            elif "#" in lines[i]:
                # matches = regex.match(lines[i])
                # if matches is None:
                #     print()
                grp1, grp2, grp3 = regex.match(lines[i]).groups()
                assert grp1 is None and grp2 is None, 'Group 1 or 2 found when matching for comments.'
                assert grp3 is not None, 'Group 3 not found when matching for comments.'
                docs.append(grp3.strip(' #') + '\n')
            elif '"""' in lines[i]:
                i += 1
                while '"""' not in lines[i]:
                    docs.append(lines[i])
                    i += 1
                docs[-1] += '\n'
            i += 1
    with open(pyfile.removesuffix('.py') + '_docs.txt', 'wt') as file:
        file.writelines(docs)


def main():
    get_docs('lensing_simulation.py')


if __name__ == '__main__':
    main()
