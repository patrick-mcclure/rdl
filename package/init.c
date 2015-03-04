#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define rdl_(NAME) TH_CONCAT_3(rdl_, Real, NAME)

#include "generic/ERDLCriterion.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libnn(lua_State *L);

int luaopen_libnn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "rdl");

  nn_FloatERDLCriterion_init(L);

  nn_DoubleERDLCriterion_init(L);
}


