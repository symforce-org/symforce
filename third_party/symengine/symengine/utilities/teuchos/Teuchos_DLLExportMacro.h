#if defined (_WIN32) && defined (BUILD_SHARED_LIBS)
#  if defined(TEUCHOS_LIB_EXPORTS_MODE)
#    define TEUCHOS_LIB_DLL_EXPORT __declspec(dllexport)
#  else
#    define TEUCHOS_LIB_DLL_EXPORT __declspec(dllimport)
#  endif
#else
#  define TEUCHOS_LIB_DLL_EXPORT
#endif
