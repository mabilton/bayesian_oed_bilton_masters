import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import approx_post
import sys
sys.path.insert(0, '.')
import plotting_helpers

def construct_true_posterior(gp, noise_cov, prior_mean, prior_cov, theta_lims=(-0.5,1.5), num_int_pts=1000):
    
    noise_icov = jnp.linalg.inv(noise_cov)
    prior_icov = jnp.linalg.inv(prior_cov)

    def unnorm_posterior(theta, y, d):
        if hasattr(gp, 'predict'):
            y_pred = gp.predict(theta, d)
        else:
            y_pred = gp(theta, d)
        unnorm_vals = -0.5*(jnp.einsum('...i,ij,...j->...', y-y_pred, noise_icov, y-y_pred) + \
                            jnp.einsum('...i,ij,...j->...', theta-prior_mean, prior_icov, theta-prior_mean))
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

def train_nn_amortised_normal_approx(gp, y_lims, d_lims, num_y_pts, num_d_pts, noise_cov, prior_mean, prior_cov, prng, num_samples, max_iter, num_layers, nn_width, activation, phi_lims, loss_name, use_reparameterisation, verbose):
    
    if 'forward' in loss_name.lower():
        loss_fun = lambda joint : approx_post.losses.ForwardKL(joint, use_reparameterisation=use_reparameterisation)
    elif 'rev' in loss_name.lower():
        loss_fun = lambda joint : approx_post.losses.ELBO(joint, use_reparameterisation=use_reparameterisation)
    else:
        raise ValueError("Invalid loss input; must choose either 'forward' or 'reverse'.")

    y_grid, d_grid = plotting_helpers.create_2d_point_grid(y_lims, d_lims, num_y_pts, num_d_pts)
    y = y_grid.flatten().reshape(-1,1)
    d = d_grid.flatten().reshape(-1,1)

    amortised_approx = _create_approx_post(prng, activation, num_layers, nn_width, phi_lims)

    model, model_grad = approx_post.models.from_surrojax_gp(gp)
    joint = approx_post.distributions.joint.ModelPlusGaussian(model, noise_cov, prior_mean, prior_cov, model_grad)   
    loss = loss_fun(joint)
    adam = approx_post.optimisers.Adam()
    loss_history = adam.fit(amortised_approx, loss, x=y, d=d, prngkey=prng, max_iter=max_iter, verbose=verbose)

    return amortised_approx, loss_history 

def load_approx_post(wts_json, activation, num_layers, width, phi_lims):
    prng = jax.random.PRNGKey(0) # Rng doesn't matter - we load weights anyways
    amortised_approx = _create_approx_post(prng, activation, num_layers, width, phi_lims)
    amortised_approx.load(wts_json)
    return amortised_approx

def _create_approx_post(prng, activation, num_layers, width, phi_lims):
    normal_approx = approx_post.distributions.approx.Gaussian(ndim=1)
    y_scale = jnp.array([0., 1.])
    d_scale = jnp.array([0., 1.])
    preprocessing = approx_post.distributions.amortised.Preprocessing.range_scaling(x=y_scale, d=d_scale)
    amortised_approx = approx_post.distributions.amortised.NeuralNetwork(normal_approx, x_dim=1, d_dim=1, prngkey=prng, preprocessing=preprocessing, activation=activation, num_layers=num_layers, width=width, phi_lims=phi_lims)
    return amortised_approx

def y_avged_kl(approx_logpdf, true_logpdf, d_theta, kind):
    approx_logpdf = jnp.array(approx_logpdf)
    true_logpdf = jnp.array(true_logpdf)
    # Forward kl:
    if 'f' in kind.lower():
        kl_vals = jnp.exp(true_logpdf)*(true_logpdf - approx_logpdf)*d_theta
    # Reverse kl:
    else:
        kl_vals = jnp.exp(approx_logpdf)*(approx_logpdf - true_logpdf)*d_theta
    return jnp.mean(jnp.sum(kl_vals, axis=1))