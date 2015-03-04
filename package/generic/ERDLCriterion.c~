#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/RSLCriterion.c"
#else

static int rdl_(RSLCriterion_updateOutput)(lua_State *L)
{
  THTensor *input1 = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *input2 = luaT_checkudata(L, 3, torch_Tensor);  
  real *RDMDistance = luaT_checkudata(L, 4, torch_Tensor);
  real sum;
  real RDMError;
  real signedError;

  sum = 0;
  TH_TENSOR_APPLY2(real, input1, real, input2,
                   real z = (*input1_data - *input2_data);
                   sum += z*z;)

  signedError = RDMDistance*RDMDistance - sum;
  
  lua_pushnumber(L, signedError);
  lua_setfield(L, 1, "signedError");
  
  RDMError = signedError * signedError;

  lua_pushnumber(L, RDMError);
  lua_setfield(L, 1, "output");
  
  lua_pushnumber(L, RDMError);
  return 1;
}

static int rdl_(RSLCriterion_updategradinput)(lua_State *L)
{
  THTensor *input1 = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *input2 = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradinput = luaT_getfieldcheckudata(L, 1, "gradinput", torch_Tensor);
  real *signedError = luaT_getfieldchecknumber(L, 1, "signedError");
  real norm =  4.;

  THTensor_(resizeAs)(gradinput, input1);
  TH_TENSOR_APPLY3(real, gradinput, real, input1, real, input2,
                   *gradinput_data = norm * (*signedError) * (*input1_data - *input2_data);)
  return 1;
}

static const struct luaL_Reg rdl_(RSLCriterion__) [] = {
  {"RSLCriterion_updateOutput", rdl_(RSLCriterion_updateOutput)},
  {"RSLCriterion_updategradinput", rdl_(RSLCriterion_updategradinput)},
  {NULL, NULL}
};

static void rdl_(RSLCriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, rdl_(RSLCriterion__), "rdl");
  lua_pop(L,1);
}

#endif
