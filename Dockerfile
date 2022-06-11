FROM dolfinx/lab:v0.4.1

RUN pip install jax==0.* \
                jaxlib==0.* \ 
                numpy==1.* \
                scipy==1.* \
                matplotlib==3.* \ 
                seaborn==0.* \
                panel==0.* \
                pyacvd==0.* \
                tetgen==0.* \
                git+https://github.com/MABilton/tetridiv@v0.0.1 \
                git+https://github.com/MABilton/arraytainers@v0.0.0 \ 
                git+https://github.com/MABilton/surrojax_gp@v0.0.0 \
                git+https://github.com/MABilton/oed_toolbox@v0.0.0 \
                git+https://github.com/MABilton/approx_post@v0.0.0

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]