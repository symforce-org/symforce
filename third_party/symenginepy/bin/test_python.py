import os
TEST_SYMPY = os.getenv('TEST_SYMPY', False)

import symengine
if not symengine.test():
    raise Exception('Tests failed')

if TEST_SYMPY:
    import sympy
    from sympy.core.cache import clear_cache
    import atexit

    atexit.register(clear_cache)
    print('Testing SYMPY')
    if not sympy.test('sympy/physics/mechanics'):
        raise Exception('Tests failed')
    if not sympy.test('sympy/liealgebras'):
        raise Exception('Tests failed')
