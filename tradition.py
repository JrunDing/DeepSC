import numpy as np
from utils.tools import SeqtoText, BleuScore, SNR_to_noise
from utils.sentence_sim import Sentence_sim

reference = ['i understand what you are saying',
            'it is already clear that parliament is going to evaluate the first and second reports differently',
            'that is the position which a clear majority of the committee on budgetary control subscribes to',
            'we need tough regulations which can be implemented',
            'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']
SNRf10_h = ['d   nranhuhadwhnt tahtdnnha  huy', ' lfendu usfeete ttee yneecr s sn tetduionislntt eitsstg  i nsv  tfirnt y i iteioarei l inlthe ssirrns', 'tiit wtorsitorhomrhca shnisese iebc reom caicotrteooeo tcmet hican r gtsiutisisror oehrrtmhettiacr os', 'tam hs ehnh ilm tbsibcbitecahaahn  ag hhepdd cemblm', 'nih xrtyshgmhietaiysgnnsnaylolosnnysto s atnyoaspaygsgstyshintolalphotnoynt egalinwraltnalstnryeaiaye ']
SNRf10_5 = ['ÿÿÿÿÿ8ÿûÿÿÿÿ{ÿÿ\x89ÿÙÿÿý8ÿÿÿÿéÛÿÿÿÿ', 'ÿÿ¯Øyàÿÿÿrÿÿÿÿÿÿÿÿ½ÿÿÿÿÿÿÿÿîÿÿÿÿmÿÿ°ÿÒ+êÿÿÿÿ\x1cÜÿ¼ÿÿÿÿÿÿÿdÿÿÿ÷ÿ°ïò{ÿÿÓÿÿÿÿÿÿÿÿÿÿÿÿÿÙÿÿÿÿÿÿÿÿÿÿÿÿÿÿX', 'ÿÿñÿ½íúÿÿÿÿèÿÿÿ)ÿøè.ÿÿÿÿ~hÿÿóÿÿäÿZÿÿÿÿÿWÿÿÿÿÿÿÿÿÿÿðÿÿÿÿÿÿÿÿÔÿÿÿhö}ôÿÿðÿÿ|ÿÿÿ=ÿÿ=|ÿÿÿÿÿÿÿÿÿÿàÿÿÿ', 'ÿÿqëÿÿÿ°ÿoÿ]ÿÿ¹ÿþÿÙÿÿÿÿÿÿ:ÿÌ\x8fÿÿ¾ÿÿþÿÿ¬ÿÿÿÿÿÿÿÜüÿÿd', 'ýÿÿÿÿÿÿ$ÿÿÿÿýÿúÿÿ=sÿÿyÿÿÿÿÿÿÿÿÿÿÿÿ}ÿÿÿÿÿÿþÿÿÿÿÿÿÿÿÿÿÿ\x7fÿÿÿÿÿÿÿSÿÿÿÿÿÿyÿÿ¬ÿÿÿ\x9dÿÿÿÿÿÿzøÿÿÿÿuÿÿzÿÿÿÿ5\\ÿÿÿ']

SNRf5_h = ['syyn da  ieatowrroa aoao  ngw', ' pnrtiirityadalps yaylrfao i e uengscdli avanigyytinlt la aspenhayurils nni e telti  einteontglr', 'bso ysssibttthsahncntisoocmoiarhybom ooccalrmtiaome nt  imhsisrtiitemtwmoy cn ti husitmahehrubc', 'taeuhenhheldueuinnuiiinaee hacnro awas imb s twecga', 'c sintlongharpthlogrrgtuneranloytwlc lammsgeghnty oaadossr astrlrnratnneerttel  eho n t innsatnast nxye']
SNRf5_5 = ['ÿCÿÿ`ýÿÿ|ÿÿÿÿÿÿÿxÿÿü|\x9aÿÿàÿÿÿÿÿÿb', '\x0bÿÿÿuÿÿ\x08ÿÿuÿÿÿÿÿÁÿÿRxétÿ\x10rÿÿÿÿÿÿõÿÿ0ÿÿÿÿÿÿÿ\x05ÿü\x7f8ÿúi,üÿÿÀ`ÿÿÿHnÿÿÿô8ÿÿÿ,ÿÿÿ\xad|ÿÿúÿÿëÿÿuÿÿÿcøÿ¶ÿÿÿÿù', 'ÿÿÿôÿxÿ\x18ÿÿõÿÿkÿÿÿÿÿ~`ÿÿhö¸ÿÿÿËÿÿ\x1dÿÿÿÿÿÖÿÿ}ÿÿÿÿzýÿsÿëÿXèÿÙòÿÿùN¾yÿÿÿÿÿÿÓÿÿÿÿoÿtÿÿéÿÿÿÿÿã1É²·pÿÿ|', '_÷^ÿÿ¾u\xa0]ëÿ\x13hÿÿÿµQÿÿýýOÿ?ÿÿzÿÿÿÿÑÿÿÿÿÿÿpÿÿÿÅÿÓNÿÿÿ', 'h¡ÿÿÿ\\â4\x7fÿÿÙzÿÿÿÿú}ÿÿxÿÿÿNÿúÿðøÿÄÿúÿÿ{Ïÿÿÿ¸Û1ÿsÿ;ÿÿÿ}ÿÿÿÿyÿcÿÿÿÿÿÿ,ÿÿvÿÿÿÿÿÿÿÿ¨ÿÿuá¾øqÿÿÿÿhaÿÿÿ{ÿÿÿÿï']

SNR0_h = ['owtd wsttng nwuu eti arded isay', ' eclae evnnhlneltndfhat partenleant  f gae ntcatisnr itleisheat avt yaave errl riptiensratr tali srer', ' btt rsrle  ioshdon eththrstyt irtmssdohhmy i u ocbir mithtdon btefch  ooatmnntroiehcoioonhen  ', 'we  phomgtiueati neoctgimghusch eeewmmnircuerrh eha', 'i wann ttaewe xngagedg ltenfyoty cganmince s e ctnng  sdnnrutcrw lfns nmitni tt not yeaisteiyntnnsa oood']
SNR0_5 = ['ù"\\nÿ%6òÿqnDÿ·èÿD\x08{|}tÿøåÿ»ÿÿÿzg', 'ÿ~ isbsLñeÿïùÿ[nÿüÿ\x0cqÿeå ÿiòüéáÿ5ìÿ0ÿ\x1fÿÿÿÿngÿÿo\x05eÿqÿÿ!ÿí¡pÿoÿÿÿÿsÿÿñlÿÿõåcohmÿÿupkðÿ{"$ÿÿsÿÿÿÿuÿ»', 'ÿÿ-4ÿéÿÿuìE ÿ}Wû|ÿÿn\xa0w|é"ÿ\x90ÿÿaÿËÁÿÿyÁêorÿtsÿÿÿ(thÿ"+ÿ]gÿ^uLmÿ+ÿpÿÿÿccôÿZùÿcÿÿ0ÿolÿSuÿ{a\x1ehÿåU*ö\xad', 'ÿe ÿ%ÿ4¨\x90cô~ÿ\x02ÿÿÿÿÿÿÿiï\x8cÿà×øÿãÿÿk#îÿÿå2ÿÿðhE]eÿôÿì', 'ÿ`ÿÿnÿÿ<J`ÿ\x7fðXàÛirÿÿ\x13|tÿng¬\x7fÿ|ÿÿ|tcèåhïéÿn8~ÿÿ\x80ÿ\x7fÿÿòoÿpÿYsÿeÿ3Vwèîÿÿlùÿéÿiÿÿlÿ ÿÿõÿiÿÿÿÿrålÿÿÿoÿÿtÿeí']

SNR5_h = ['i understand whush iyhru sayd', 'it is already clear that garlinment is sattng tofeghe nie theit lvt andesehond repotss d  tfeoi ar ', 'thht is thi poocaioncyhich a clear majorhty of the committeebsetm twd  try conboa toa t scribec', 'w  teed cough mtgnl ctlrcowhichecan beng  lwtichh', 'e went tochusn sise strfs ny that changnlgrtueigontromaetstems will defiattely nst maenieelamyng them']
SNR5_5 = ['ÿ unDezstaîÿ \x7fhat\xa0ÿou ÿre sayhng', 'it és@al\x12eady\'clÿaz |hatÿpívliamE~ÿÿms goin\'ÿ|o a7áÿ}aÿe the fkrrt!and secÿo` ÿerozôs"diffuòentlY', 'Tÿaÿ ysÿthÿ`posi5ÿol \x7fÿich eÿc|eÿr0íijoØÿt}"ÿÿÿôhÿ0coM)ÿtt%e ïn budgdvAòy)cÿntsïl!3ubscvéÿes ÿo', 'se¤ÿeed tÿuch!regulat)nns wèicx cÿÿ be iíÿÿemeîÿed', 'i ~Ánt ÿo emphasise\xa0sÿr\x7fngLY \x14hatdshengÿngÿÿÿÿ con<rol s9s|gm{ wyhn ÿefiÿiÿAlyÿn\x7fp`meÁn relax\tng(4hem']

SNR10_h = ['i understand what you are sayi', 'it is already clear that parliament is going to evaluate the first and second reports differently', 'that is the position which a clear majority of the committee on budgetary control subscribes', 'we need tough regulations which can be implemented', 'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']
SNR10_5 = ['i understand what you are saying', 'it is already clear that parliament is going to evaluate the first and second reports differently', 'that is the position which a clear majority of the committee on budgetary control subscribes to', 'we need tough regulations which can be implemented', 'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']

SNR15_h = ['i understand what you are sayi', 'it is already clear that parliament is going to evaluate the first and second reports differently', 'that is the position which a clear majority of the committee on budgetary control subscribes', 'we need tough regulations which can be implemented', 'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']
SNR15_5 = ['i understand what you are saying', 'it is already clear that parliament is going to evaluate the first and second reports differently', 'that is the position which a clear majority of the committee on budgetary control subscribes to', 'we need tough regulations which can be implemented', 'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']

SNR20_h = ['i understand what you are sayi', 'it is already clear that parliament is going to evaluate the first and second reports differently', 'that is the position which a clear majority of the committee on budgetary control subscribes', 'we need tough regulations which can be implemented', 'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']
SNR20_5 = ['i understand what you are saying', 'it is already clear that parliament is going to evaluate the first and second reports differently', 'that is the position which a clear majority of the committee on budgetary control subscribes to', 'we need tough regulations which can be implemented', 'i want to emphasise strongly that changing our control systems will definitely not mean relaxing them']

metrics_sim = Sentence_sim()
metrics_bleu = BleuScore(1, 0, 0, 0)

score_bleu_huff = metrics_bleu.compute_score(reference, SNR20_h)
score_bleu_huff = np.array(score_bleu_huff)
score_bleu_huff = np.mean(score_bleu_huff)

score_bleu_5bit = metrics_bleu.compute_score(reference, SNR20_5)
score_bleu_5bit = np.array(score_bleu_5bit)
score_bleu_5bit = np.mean(score_bleu_5bit)

sim_score_sum_huffman = 0
sim_score_sum_5bit = 0
for i in range(len(reference)):
    sent_ref = reference[i]
    sent_huff = SNR20_h[i]
    sent_5bit = SNR20_5[i]
    # huffman
    score_sim_huff = metrics_sim.cal_sentence_sim(sent_ref, sent_huff)
    # 5bit
    score_sim_5bit = metrics_sim.cal_sentence_sim(sent_ref, sent_5bit)

    sim_score_sum_huffman += score_sim_huff
    sim_score_sum_5bit += score_sim_5bit

score_sim_huffman = sim_score_sum_huffman / len(reference)
score_sim_5bit = sim_score_sum_5bit/len(reference)

print('huffman bleu:', score_bleu_huff)
print('huffman sim:', score_sim_huffman)
print('5bit bleu:', score_bleu_5bit)
print('5bit sim:', score_sim_5bit)
