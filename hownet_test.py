import OpenHowNet
from utility import hownet
h=hownet()
h.sys_possible(['I','am','from','China']) 
hownet_dict = OpenHowNet.HowNetDict()
# Get all the senses represented by the word "苹果".

en_words_list=hownet_dict.get_en_words()
'mainland China' in en_words_list


words=hownet_dict.get_en_words()
hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)
sys=hownet_dict_advanced.get_nearest_words('give up', language='en',K=1)
