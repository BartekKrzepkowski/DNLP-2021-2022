get_ipython().getoutput("python3 sanity_check.py 1d")


get_ipython().getoutput("python3 sanity_check.py 1e")


get_ipython().getoutput("python3 sanity_check.py 1f")


get_ipython().getoutput("sh run.sh vocab")


get_ipython().getoutput("sh run.sh train_local")


get_ipython().getoutput("sh run.sh train")


get_ipython().getoutput("sh run.sh test")


BLUE = 10.653129136263704

print(f'Corpus BLEU: {BLUE:.4f}')





get_ipython().getoutput("sh run.sh test_full")


BLUE_full = 0.19279697082418473

print(f'Corpus BLEU: {BLUE_full:.4f}')



