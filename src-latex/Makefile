name = main-paper
TEX = latexmk --pdf -pdflatex="pdflatex  --shell-escape"
INPUT_FILE = $(name).tex
DIST_FILES = orcidlink.sty Makefile figs images splncs04.bst llncs.cls $(name).tex refs.bib $(name).bbl

.PHONY: all
all:
	$(TEX) $(INPUT_FILE) -jobname="$(name)"

.PHONY: final
final: all
	ghostscript -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(name).final.pdf $(name).pdf

.PHONY: clean
clean:
	latexmk -C $(INPUT_FILE) -jobname="$(name)"
	rm -f $(name).final.pdf
	rm -f $(name).zip

.PHONY: dist
dist: final
	git clean -xi figs images
	zip -r $(name).zip $(DIST_FILES)

