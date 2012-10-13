

.PHONY: active

active:
	#export STD_SIMS='NEURON'; export STD_SCENS='02[01]'; waf cleanup generate compare
	#export STD_SIMS='NEURON;morphforge;mfcuke'; export STD_SCENS='020'; waf cleanup generate compare
	export STD_SIMS='NEURON;morphforge;mfcuke'; export STD_SCENS='020'; waf cleanup generate compare


all:
	export STD_SIMS='*'; export STD_SCENS='*'; waf cleanup generate compare

clean:
	find . -name 'output' -type l -exec rm {} \;
	find . -name 'parser.out' -type l -exec rm {} \;
	rm -rf output/
	


edit:
	find scenario_descriptions/ | xargs geany &
	find simulators  -type f| xargs geany &

