import sys


def main(input_name, output_name, replacements):
    replacements = dict((item.split("=")[0], item.split("=")[1] == "True") for item in replacements)
    with open(input_name, "r") as inp:
        text = inp.readlines()

    new_text = []
    in_cond = [True]
    nspaces = [0]
    for i, line in enumerate(text):
        if line.strip().startswith("IF"):
            s = len(line) - len(line.lstrip())
            while s <= nspaces[-1] and len(in_cond) > 1:
                in_cond = in_cond[:-1]
                nspaces = nspaces[:-1]

            cond = line.lstrip()[3:-2]
            in_cond.append(replacements[cond])
            nspaces.append(s)
        elif line.strip().startswith("ELSE"):
            in_cond[-1] = not in_cond[-1] and in_cond[-2]

        if len(line) > 1 and not line.strip().startswith(("IF", "ELSE")):
            while len(in_cond) > 1 and (len(line) <= nspaces[-1] or not line.startswith(" "*nspaces[-1]) or line[nspaces[-1]] != " "):
                in_cond = in_cond[:-1]
                nspaces = nspaces[:-1]
        if len(line) == 1:
            new_text.append(line)
        elif in_cond[-1] and not line.strip().startswith(("IF", "ELSE")):
            new_text.append(line[4*(len(in_cond) - 1):])
    
    with open(output_name, "w") as out:
        out.writelines(new_text)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
