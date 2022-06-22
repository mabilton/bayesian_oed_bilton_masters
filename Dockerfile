FROM dolfinx/lab:v0.4.1

RUN pip install "jax>=0.2,<=0.2.25" \
                "jaxlib~=0.1,<=0.1.75" \ 
                numpy~=1.20 \
                scipy~=1.7 \
                matplotlib~=3.4 \ 
                seaborn~=0.11 \
                panel~=0.13 \
                pyacvd~=0.2 \
                tetgen~=0.6 \
                numpyencoder~=0.3 \
                git+https://github.com/MABilton/tetridiv@v0.0.1 \
                git+https://github.com/MABilton/arraytainers@v0.0.0 \ 
                git+https://github.com/MABilton/surrojax_gp@v0.0.1 \
                git+https://github.com/MABilton/oed_toolbox@v0.0.0 \
                git+https://github.com/MABilton/approx_post@v0.0.0

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]