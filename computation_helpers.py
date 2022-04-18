import jax.numpy as jnp
import jax.scipy as jscipy
import approx_post
from . import plotting_helpers

def compute_true_log_posterior(model, noise_cov, prior_mean, prior_cov, theta_lims=(-0.5,1.5), num_int_pts=1000):
    
    noise_icov = jnp.linalg.inv(noise_cov)
    prior_icov = jnp.linalg.inv(prior_cov)

    def unnorm_posterior(theta, y, d):
        y_pred = model.predict(theta, d)
        unnorm_vals = -0.5*(jnp.einsum('ai,ij,aj->a', y-y_pred, noise_icov, y-y_pred) + \
                            jnp.einsum('ai,ij,aj->a', theta-prior_mean, prior_icov, theta-prior_mean))
        return jnp.exp(unnorm_vals)

    theta_int = jnp.linspace(*theta_lims, num_int_pts).reshape(-1,1)
    d_theta = (theta_lims[1]-theta_lims[0])/(num_int_pts-1)
    def norm_const(y, d):
        unnorm_vals = unnorm_posterior(theta_int, y, d)
        return jnp.trapz(unnorm_vals.squeeze(), dx=d_theta)

    def true_posterior(theta, y, d):
        norm_const_val = norm_const(y, d)
        unnorm_val = unnorm_posterior(theta, y, d)
        return unnorm_val/norm_const_val

    return true_posterior

def train_amortised_normal_approx(gp, y_vals, d_vals, noise_cov, prior_mean, prior_cov, prng, loss, num_samples):
    
    if 'forward' in loss.lower():
        loss_fun = lambda : joint approx_post.losses.ForwardKL(joint, use_reparameterisation=False)
    elif 'rev' in loss.lower():
        loss_fun = lambda : joint approx_post.losses.ELBO(joint, use_reparameterisation=False)
    else:
        raise ValueError("Invalid loss input; must choose either 'forward' or 'reverse'.")

    def model(theta, x):
        d = 
        x_val = jnp.stack([theta, d], axis=0)
        return gp.predict(x_val)['mean']
    model_grad = jax.jacfwd(model, argnums=0)
    
    joint = approx_post.joint.ModelPlusNoise(model, noise_cov, prior_mean, prior_cov, model_grad)    
    normal_approx = approx_post.approx.Gaussian(ndim=1)
    amortised_approx = approx_post.amortised.NeuralNetwork(normal_approx, x_dim=2, prngkey=prng)

    loss = loss_fun(jointdist)
    adam = approx_post.optimisers.Adam()
    loss_history = adam(approx, elbo_loss, x, prng)

    return approx, loss_history 

def compute_forward_kl(p, q):
    return jnp.sum(p*jnp.log(p/q))

def compute_reverse_kl(p, q):
    return jnp.sum(q*jnp.log(q/p))