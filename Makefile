
clean:
	find . -name 'output' -type l -exec rm {} \;
	find . -name 'parser.out' -type l -exec rm {} \;
	rm -rf output/
	


edit:
	find scenario_descriptions/ | xargs geany &
	find simulators  -type f| xargs geany &
