from distutils.sysconfig import get_config_var
extsuffix = get_config_var('EXT_SUFFIX')
if extsuffix is None:
    print("")
else:
    print(extsuffix[0:].rsplit(".", 1)[0])
