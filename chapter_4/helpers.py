import jax.numpy as jnp

def compute_true_log_posterior(y, d, model, noise_cov, prior_mean, prior_cov):

    def unnorm_true_log_posterior(theta, y, d, y_pred):
        return -0.5*np.einsum('ai,ij,aj->a', y-y_pred, noise_cov, y-y_pred) + \
               -0.5*np.einsum('ai,ij,aj->a', theta-prior_mean, prior_cov, theta-prior_mean)
    
    def unnorm_true_log_posterior(theta, y, d):
        y_pred = model(theta, d)
        unnorm_post = 
        norm_const = jnp.trapz(unnorm_post, dx=0.01)
        return unnorm_post/norm_const


    return true_posterior

def compute_laplace_approx_mean_and_cov():
    

def compute_forward_kl_divergence():