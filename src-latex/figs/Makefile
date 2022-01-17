####################################################################
##
## Compile every ".tex" standalone tikz files into SVG or PDF
## Every file ending with .inc.tex will be ignored (this can be used to ignore headers tex file)
##
COMPILE_SVG =     # Set to "" if you don't want to compile into SVGs
COMPILE_PNG =     # Set to "" if you don't want to compile into PNGs
COMPILE_PDF = ON  # Set to "" if you don't want to compile into PDFS



####################################################################

TEX_FILES  := $(shell find . -name '*.tex' -a -not -name '*.inc.tex')
PLOT_FILES := $(shell find . -name '*.plot.py')
MMD_FILES  := $(shell find . -name '*.mmd')

SVG_FILES = $(and ${COMPILE_SVG}, ${TEX_FILES:%.tex=%.svg})
PNG_FILES = $(and ${COMPILE_PNG}, ${TEX_FILES:%.tex=%.png})
PDF_FILES = $(and ${COMPILE_PDF}, ${TEX_FILES:%.tex=%.pdf})

SVG_FILES += $(and ${COMPILE_SVG}, ${PLOT_FILES:%.py=%.svg})
PNG_FILES += $(and ${COMPILE_PNG}, ${PLOT_FILES:%.py=%.png})
PDF_FILES += $(and ${COMPILE_PDF}, ${PLOT_FILES:%.py=%.pdf})

SVG_FILES += $(and ${COMPILE_SVG}, ${MMD_FILES:%.mmd=%.svg})
PNG_FILES += $(and ${COMPILE_PNG}, ${MMD_FILES:%.mmd=%.png})
PDF_FILES += $(and ${COMPILE_PDF}, ${MMD_FILES:%.mmd=%.pdf})



all: ${SVG_FILES} ${PNG_FILES} ${PDF_FILES}

%.pdf: %.mmd
	mmdc -i $< -o $@
	pdfcrop $@ $@

%.svg: %.mmd
	mmdc -i $< -o $@

%.png: %.mmd
	mmdc -i $< -o $@

%.pdf: %.tex
	latexmk -pdf $< 

%.svg: %.pdf
	pdf2svg $< $@

%.png: %.pdf
	#FIXME

%.plot.png: %.plot.py
	python3 $< $@

%.plot.pdf: %.plot.py
	python3 $< $@

%.plot.svg: %.plot.py
	python3 $< $@
