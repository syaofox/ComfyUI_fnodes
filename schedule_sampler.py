import numpy as np
import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from nodes import common_ksampler

_CATEGORY = 'fnodes/Samplers'

NOISE_LEVELS = {
    'SD1': [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904, 0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582],
    'SDXL': [14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945, 0.8620721141, 0.5550693289, 0.3798540708, 0.2332364134, 0.1114188177, 0.0291671582],
    'SVD': [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002],
}


def common_sampling_logic(model, add_noise, noise_seed, cfg, positive, negative, sampler_name, steps, latent_image, sigmas):
    sampler = comfy.samplers.sampler_object(sampler_name)

    if isinstance(latent_image, dict) and 'samples' in latent_image:
        latent = latent_image.copy()
        latent_samples = latent_image['samples']
    else:
        latent = {'samples': latent_image}
        latent_samples = latent_image

    latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
    latent['samples'] = latent_samples

    if not add_noise:
        noise = Noise_EmptyNoise().generate_noise(latent)
    else:
        noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

    noise_mask = latent.get('noise_mask')

    x0_output = {}
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_samples,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=noise_seed,
    )

    out = {'samples': samples}
    if 'x0' in x0_output:
        out_denoised = {'samples': model.model.process_latent_out(x0_output['x0'].cpu())}
    else:
        out_denoised = out

    return out, out_denoised


class ScheduleSamplerCustomTurbo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'add_noise': ('BOOLEAN', {'default': True}),
                'noise_seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF},
                ),
                'cfg': (
                    'FLOAT',
                    {
                        'default': 8.0,
                        'min': 0.0,
                        'max': 100.0,
                        'step': 0.1,
                        'round': 0.01,
                    },
                ),
                'positive': ('CONDITIONING',),
                'negative': ('CONDITIONING',),
                'sampler_name': (comfy.samplers.SAMPLER_NAMES,),
                'steps': ('INT', {'default': 4, 'min': 1, 'max': 10}),
                'denoise_schedule': ('STRING', {'default': '0.5,0.25'}),
                'latent_image': ('LATENT',),
            },
        }

    RETURN_TYPES = ('LATENT', 'LATENT')
    RETURN_NAMES = ('output', 'denoised_output')

    FUNCTION = 'sample'

    CATEGORY = _CATEGORY

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler_name, steps, denoise_schedule, latent_image):
        denoise_values = [float(x.strip()) for x in denoise_schedule.split(',')]

        mask = latent_image.get('noise_mask', None)

        for i, denoise in enumerate(denoise_values):
            current_noise_seed = noise_seed + i
            start_step = 10 - int(10 * denoise)
            timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step : start_step + steps]
            sigmas = model.get_model_object('model_sampling').sigma(timesteps)
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])

            latent_image['noise_mask'] = mask
            latent_image, out_denoised = common_sampling_logic(model, add_noise, current_noise_seed, cfg, positive, negative, sampler_name, steps, latent_image, sigmas)

        return latent_image, out_denoised


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


class ScheduleSamplerCustomAYS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'add_noise': ('BOOLEAN', {'default': True}),
                'noise_seed': (
                    'INT',
                    {'default': 0, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF},
                ),
                'cfg': (
                    'FLOAT',
                    {
                        'default': 8.0,
                        'min': 0.0,
                        'max': 100.0,
                        'step': 0.1,
                        'round': 0.01,
                    },
                ),
                'positive': ('CONDITIONING',),
                'negative': ('CONDITIONING',),
                'sampler_name': (comfy.samplers.SAMPLER_NAMES,),
                'model_type': (['SD1', 'SDXL', 'SVD'],),
                'steps': ('INT', {'default': 10, 'min': 10, 'max': 10000}),
                'denoise_schedule': ('STRING', {'default': '0.5,0.25'}),
                'latent_image': ('LATENT',),
            }
        }

    RETURN_TYPES = ('LATENT', 'LATENT')
    RETURN_NAMES = ('output', 'denoised_output')

    FUNCTION = 'sample'

    CATEGORY = _CATEGORY

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler_name, model_type, steps, denoise_schedule, latent_image):
        denoise_values = [float(x.strip()) for x in denoise_schedule.split(',')]
        mask = latent_image.get('noise_mask', None)

        for i, denoise in enumerate(denoise_values):
            current_noise_seed = noise_seed + i

            total_steps = steps
            if denoise < 1.0:
                if denoise <= 0.0:
                    return (torch.FloatTensor([]),)
                total_steps = round(steps * denoise)

            sigmas = NOISE_LEVELS[model_type][:]
            if (steps + 1) != len(sigmas):
                sigmas = loglinear_interp(sigmas, steps + 1)

            sigmas = sigmas[-(total_steps + 1) :]
            sigmas[-1] = 0
            sigmas = torch.FloatTensor(sigmas)

            latent_image['noise_mask'] = mask
            latent_image, out_denoised = common_sampling_logic(model, add_noise, current_noise_seed, cfg, positive, negative, sampler_name, total_steps, latent_image, sigmas)

        return latent_image, out_denoised


class ScheduleSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'The model used for denoising the input latent.'}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF, 'tooltip': 'The random seed used for creating the noise.'}),
                'steps': ('INT', {'default': 20, 'min': 1, 'max': 10000, 'tooltip': 'The number of steps used in the denoising process.'}),
                'cfg': (
                    'FLOAT',
                    {
                        'default': 8.0,
                        'min': 0.0,
                        'max': 100.0,
                        'step': 0.1,
                        'round': 0.01,
                        'tooltip': 'The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.',
                    },
                ),
                'sampler_name': (comfy.samplers.KSampler.SAMPLERS, {'tooltip': 'The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.'}),
                'scheduler': (comfy.samplers.KSampler.SCHEDULERS, {'tooltip': 'The scheduler controls how noise is gradually removed to form the image.'}),
                'positive': ('CONDITIONING', {'tooltip': 'The conditioning describing the attributes you want to include in the image.'}),
                'negative': ('CONDITIONING', {'tooltip': 'The conditioning describing the attributes you want to exclude from the image.'}),
                'latent_image': ('LATENT', {'tooltip': 'The latent image to denoise.'}),
                'denoise_schedule': ('STRING', {'default': '0.5,0.25'}),
            }
        }

    RETURN_TYPES = ('LATENT',)
    OUTPUT_TOOLTIPS = ('The denoised latent.',)
    FUNCTION = 'sample'

    CATEGORY = _CATEGORY

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise_schedule):
        denoise_values = [float(x.strip()) for x in denoise_schedule.split(',')]

        mask = latent_image.get('noise_mask', None)
        for i, denoise in enumerate(denoise_values):
            current_noise_seed = seed + i
            current_steps = round(steps * denoise)

            latent_image['noise_mask'] = mask
            latent_image = common_ksampler(model, current_noise_seed, current_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            latent_image = latent_image[0]

        return (latent_image,)


SAMPLER_CLASS_MAPPINGS = {
    'ScheduleSamplerCustomTurbo-': ScheduleSamplerCustomTurbo,
    'ScheduleSamplerCustomAYS-': ScheduleSamplerCustomAYS,
    'ScheduleSampler-': ScheduleSampler,
}

SAMPLER_NAME_MAPPINGS = {
    'ScheduleSamplerCustomTurbo-': 'ScheduleSamplerCustomTurbo',
    'ScheduleSamplerCustomAYS-': 'ScheduleSamplerCustomAYS',
    'ScheduleSampler-': 'ScheduleSampler',
}
