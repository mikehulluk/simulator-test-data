

.PHONY: active

active:
	#export STD_SIMS='NEURON'; export STD_SCENS='02[01]'; waf cleanup generate compare
	#export STD_SIMS='NEURON;morphforge;mfcuke'; export STD_SCENS='020'; waf cleanup generate compare
	#export STD_SIMS='NEURON;morphforge;mfcuke'; export STD_SCENS='022'; export STD_SHORT='TRUE';  waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='022'; export STD_SHORT='TRUE';  waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='022';   waf cleanup generate compare
	#export STD_SIMS='NEURON'; export STD_SCENS='030';   waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='03[5]'; export STD_SHORT='TRUE';   waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='03[5]'; waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='03[5]'; export STD_SHORT='TRUE'; waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='*'; export STD_SHORT='TRUE'; waf cleanup generate compare
	#export STD_SIMS='morphforge'; export STD_SCENS='001'; export STD_SHORT='TRUE'; waf all
	#export STD_SIMS='NEURON'; export STD_SCENS='001'; export STD_SHORT='TRUE'; waf all
	#export STD_SIMS='morphforge'; export STD_SCENS='020'; waf compare
	#export STD_SIMS='*'; export STD_SCENS='*';  waf compare
	#export STD_SIMS='*'; export STD_SCENS='*';  waf compare
	#export STD_SIMS='morphforge' export STD_SCENS='022';  waf generate compare
	#export STD_SIMS='*' export STD_SCENS='*';  waf compare
	export STD_SIMS='morphforge' export STD_SCENS='021';  waf generate 
	export STD_SIMS='*' export STD_SCENS='*';  waf compare
	#export STD_SIMS='*'; export STD_SCENS='001'; export STD_SHORT='TRUE'; waf all
	


all:
	export STD_SIMS='*'; export STD_SCENS='*'; waf cleanup generate compare

clean:
	find . -name 'output' -type l -exec rm {} \;
	find . -name 'parser.out'  -exec rm {} \;
	rm -rf output/* _output/
	find . -name '*.svg'  -exec rm {} \;
	find . -name '*.png'  -exec rm {} \;
	waf cleanup
	


edit:
	find scenario_descriptions/ -type f| xargs geany &
	find simulators  -type f | xargs geany &
	find src/mfcuke/  -type f| xargs geany &
	find src/simtest_utils/  -type f | xargs geany &
	find src/waf_util/  -type f | xargs geany &

