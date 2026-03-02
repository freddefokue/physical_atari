import re

log_text = """
0:ppo frame:  390298  503/s eps 122, 3056= -21 u 190 ploss -0.007 vloss 0.002 ent 1.204 kl 0.006 avg -21.0
0:ppo frame:  393354  432/s eps 123, 3056= -21 u 192 ploss -0.005 vloss 0.001 ent 1.207 kl 0.003 avg -21.0
0:ppo frame:  396410  531/s eps 124, 3056= -21 u 193 ploss -0.009 vloss 0.002 ent 1.254 kl 0.009 avg -21.0
0:ppo frame:  399466  553/s eps 125, 3056= -21 u 194 ploss -0.009 vloss 0.002 ent 1.217 kl 0.010 avg -21.0
0:ppo frame:  402522  430/s eps 126, 3056= -21 u 196 ploss -0.011 vloss 0.002 ent 1.240 kl 0.010 avg -21.0
0:ppo frame:  405578  553/s eps 127, 3056= -21 u 197 ploss -0.010 vloss 0.002 ent 1.139 kl 0.005 avg -21.0
0:ppo frame:  408634  428/s eps 128, 3056= -21 u 199 ploss -0.010 vloss 0.002 ent 1.116 kl 0.001 avg -21.0
0:ppo frame:  411690  536/s eps 129, 3056= -21 u 200 ploss -0.009 vloss 0.002 ent 1.011 kl 0.009 avg -21.0
0:ppo frame:  414746  424/s eps 130, 3056= -21 u 202 ploss -0.007 vloss 0.002 ent 1.201 kl 0.002 avg -21.0
0:ppo frame:  417802  547/s eps 131, 3056= -21 u 203 ploss -0.010 vloss 0.002 ent 1.071 kl 0.001 avg -21.0
0:ppo frame:  420858  435/s eps 132, 3056= -21 u 205 ploss -0.008 vloss 0.002 ent 0.946 kl 0.005 avg -21.0
0:ppo frame:  423914  501/s eps 133, 3056= -21 u 206 ploss -0.010 vloss 0.002 ent 0.793 kl 0.010 avg -21.0
0:ppo frame:  426970  401/s eps 134, 3056= -21 u 208 ploss -0.009 vloss 0.002 ent 0.978 kl 0.006 avg -21.0
0:ppo frame:  430026  539/s eps 135, 3056= -21 u 209 ploss -0.008 vloss 0.002 ent 1.076 kl 0.004 avg -21.0
0:ppo frame:  433082  428/s eps 136, 3056= -21 u 211 ploss -0.013 vloss 0.002 ent 1.077 kl 0.008 avg -21.0
0:ppo frame:  436138  533/s eps 137, 3056= -21 u 212 ploss -0.011 vloss 0.002 ent 1.263 kl 0.009 avg -21.0
0:ppo frame:  439194  426/s eps 138, 3056= -21 u 214 ploss -0.009 vloss 0.002 ent 1.132 kl 0.004 avg -21.0
0:ppo frame:  442250  549/s eps 139, 3056= -21 u 215 ploss -0.010 vloss 0.002 ent 1.146 kl 0.006 avg -21.0
0:ppo frame:  445306  434/s eps 140, 3056= -21 u 217 ploss -0.012 vloss 0.001 ent 1.056 kl 0.007 avg -21.0
0:ppo frame:  448362  538/s eps 141, 3056= -21 u 218 ploss -0.007 vloss 0.002 ent 1.148 kl 0.006 avg -21.0
0:ppo frame:  451418  420/s eps 142, 3056= -21 u 220 ploss -0.010 vloss 0.002 ent 1.327 kl 0.012 avg -21.0
0:ppo frame:  454474  544/s eps 143, 3056= -21 u 221 ploss -0.009 vloss 0.002 ent 1.252 kl 0.004 avg -21.0
0:ppo frame:  457530  433/s eps 144, 3056= -21 u 223 ploss -0.008 vloss 0.002 ent 1.182 kl 0.004 avg -21.0
0:ppo frame:  460586  565/s eps 145, 3056= -21 u 224 ploss -0.008 vloss 0.002 ent 1.099 kl 0.010 avg -21.0
0:ppo frame:  463642  430/s eps 146, 3056= -21 u 226 ploss -0.008 vloss 0.002 ent 1.266 kl 0.007 avg -21.0
0:ppo frame:  466698  504/s eps 147, 3056= -21 u 227 ploss -0.007 vloss 0.003 ent 1.286 kl 0.005 avg -21.0
0:ppo frame:  469754  409/s eps 148, 3056= -21 u 229 ploss -0.013 vloss 0.002 ent 1.090 kl 0.008 avg -21.0
0:ppo frame:  472810  535/s eps 149, 3056= -21 u 230 ploss -0.010 vloss 0.003 ent 1.093 kl 0.003 avg -21.0
0:ppo frame:  475866  432/s eps 150, 3056= -21 u 232 ploss -0.004 vloss 0.002 ent 1.359 kl 0.004 avg -21.0
0:ppo frame:  478922  544/s eps 151, 3056= -21 u 233 ploss -0.009 vloss 0.003 ent 1.298 kl 0.007 avg -21.0
0:ppo frame:  481978  420/s eps 152, 3056= -21 u 235 ploss -0.009 vloss 0.002 ent 1.253 kl 0.008 avg -21.0
0:ppo frame:  485034  524/s eps 153, 3056= -21 u 236 ploss -0.009 vloss 0.003 ent 1.153 kl 0.008 avg -21.0
0:ppo frame:  488090  433/s eps 154, 3056= -21 u 238 ploss -0.010 vloss 0.002 ent 0.775 kl 0.008 avg -21.0
0:ppo frame:  491146  548/s eps 155, 3056= -21 u 239 ploss -0.008 vloss 0.003 ent 1.184 kl 0.004 avg -21.0
0:ppo frame:  494202  431/s eps 156, 3056= -21 u 241 ploss -0.010 vloss 0.002 ent 1.166 kl 0.011 avg -21.0
0:ppo frame:  497258  544/s eps 157, 3056= -21 u 242 ploss -0.016 vloss 0.003 ent 1.176 kl 0.006 avg -21.0
0:ppo frame:  500314  425/s eps 158, 3056= -21 u 244 ploss -0.006 vloss 0.003 ent 1.137 kl 0.006 avg -21.0
0:ppo frame:  503370  546/s eps 159, 3056= -21 u 245 ploss -0.013 vloss 0.003 ent 1.317 kl 0.005 avg -21.0
0:ppo frame:  506426  431/s eps 160, 3056= -21 u 247 ploss -0.003 vloss 0.003 ent 1.167 kl 0.001 avg -21.0
0:ppo frame:  509482  550/s eps 161, 3056= -21 u 248 ploss -0.010 vloss 0.002 ent 1.008 kl 0.000 avg -21.0
0:ppo frame:  512538  431/s eps 162, 3056= -21 u 250 ploss -0.010 vloss 0.002 ent 1.175 kl 0.005 avg -21.0
0:ppo frame:  515594  539/s eps 163, 3056= -21 u 251 ploss -0.010 vloss 0.003 ent 0.876 kl 0.006 avg -21.0
0:ppo frame:  518650  394/s eps 164, 3056= -21 u 253 ploss -0.010 vloss 0.002 ent 1.072 kl 0.010 avg -21.0
0:ppo frame:  521706  488/s eps 165, 3056= -21 u 254 ploss -0.012 vloss 0.002 ent 1.249 kl 0.008 avg -21.0
0:ppo frame:  524762  397/s eps 166, 3056= -21 u 256 ploss -0.016 vloss 0.001 ent 1.136 kl 0.012 avg -21.0
0:ppo frame:  527818  514/s eps 167, 3056= -21 u 257 ploss -0.012 vloss 0.002 ent 1.252 kl 0.004 avg -21.0
0:ppo frame:  530874  419/s eps 168, 3056= -21 u 259 ploss -0.006 vloss 0.002 ent 1.088 kl 0.006 avg -21.0
0:ppo frame:  533930  522/s eps 169, 3056= -21 u 260 ploss -0.011 vloss 0.002 ent 1.221 kl 0.009 avg -21.0
0:ppo frame:  536986  414/s eps 170, 3056= -21 u 262 ploss -0.010 vloss 0.002 ent 1.403 kl 0.012 avg -21.0
0:ppo frame:  540042  523/s eps 171, 3056= -21 u 263 ploss -0.014 vloss 0.001 ent 0.970 kl 0.006 avg -21.0
0:ppo frame:  543098  415/s eps 172, 3056= -21 u 265 ploss -0.013 vloss 0.001 ent 1.177 kl 0.008 avg -21.0
0:ppo frame:  546154  523/s eps 173, 3056= -21 u 266 ploss -0.012 vloss 0.002 ent 1.268 kl 0.009 avg -21.0
0:ppo frame:  549210  427/s eps 174, 3056= -21 u 268 ploss -0.010 vloss 0.001 ent 1.239 kl 0.009 avg -21.0
"""
ents = []
for line in log_text.strip().split("\n"):
    m = re.search(r"ent\s+([0-9.]+)", line)
    if m:
        ents.append(float(m.group(1)))

print(f"Entropy min: {min(ents):.3f}, max: {max(ents):.3f}, mean: {sum(ents)/len(ents):.3f}")
