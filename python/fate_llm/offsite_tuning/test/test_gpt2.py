from fate_llm.model_zoo.offsite_tuning.gpt2 import GPT2LMHeadMainModel, GPT2LMHeadSubModel

main_model = GPT2LMHeadMainModel('gpt2', emulator_layer_num=4, adapter_top_layer_num=2, adapter_bottom_layer_num=2)
sub_model = GPT2LMHeadSubModel('gpt2', emulator_layer_num=4, adapter_top_layer_num=2, adapter_bottom_layer_num=2)

