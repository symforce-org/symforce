# aclint: py3
import argh


def main(lcmtypes_dir: str) -> None:
    import skymarshal.package_map

    package_map = skymarshal.package_map.parse_lcmtypes([lcmtypes_dir])
    for package, types in package_map.items():
        for type_name in types.type_definitions.keys():
            print(package, type_name)


if __name__ == "__main__":
    argh.dispatch_command(main)
