#!/usr/bin/env python

import essentia
import essentia.standard as standard
import essentia.streaming as streaming

import numpy
from postprocess import postProcess

def tonalPoolCleaning(pool, namespace=None):
    tonalspace = 'tonal.'
    if namespace: tonalspace = namespace + '.tonal.'
    tuningFreq = pool[tonalspace + 'tuning_frequency'][-1]
    pool.remove(tonalspace + 'tuning_frequency')
    pool.set(tonalspace + 'tuning_frequency', tuningFreq)
    pool.remove(tonalspace + 'hpcp_highres')

def normalize(array):
    max = numpy.max(array)
    return [float(val)/float(max) for val in array]


def tuningSystemFeatures(pool, namespace=''):
    # expects tonal descriptors and tuning features to be in pool
    tonalspace = 'tonal.'
    if namespace: tonalspace = namespace + '.tonal.'

    # 1-diatonic strength
    hpcp_highres = normalize(numpy.mean(pool[tonalspace + 'hpcp_highres'], 0))
    key,scale,strength,_ = standard.Key(profileType='diatonic')(hpcp_highres)
    pool.set(tonalspace + 'tuning_diatonic_strength', strength)

    # 2- high resolution features
    eqTempDeviation, ntEnergy,_ = standard.HighResolutionFeatures()(hpcp_highres)
    pool.set(tonalspace+'tuning_equal_tempered_deviation', eqTempDeviation)
    pool.set(tonalspace+'tuning_nontempered_energy_ratio', ntEnergy)

    # 3- THPCP
    hpcp = normalize(numpy.mean(pool[tonalspace + 'hpcp'], 0))
    hpcp_copy = hpcp[:]
    idx = numpy.argmax(hpcp)
    offset = len(hpcp)-idx
    hpcp[:offset] = hpcp_copy[idx:offset+idx]
    hpcp[offset:offset+idx] = hpcp_copy[0:idx]
    pool.set(tonalspace+'thpcp', essentia.array(hpcp))


def sfxPitch(pool, namespace=''):
    sfxspace = 'sfx.'
    llspace = 'lowlevel.'
    if namespace:
        sfxspace = namespace + '.sfx.'
        llspace = namespace + '.lowlevel.'
    pitch = pool[llspace+'pitch']
    gen = streaming.VectorInput(pitch)
    maxtt = streaming.MaxToTotal()
    mintt = streaming.MinToTotal()
    amt = streaming.AfterMaxToBeforeMaxEnergyRatio()
    gen.data >> maxtt.envelope
    gen.data >> mintt.envelope
    gen.data >> amt.pitch
    maxtt.maxToTotal >> (pool, sfxspace+'pitch_max_to_total')
    mintt.minToTotal >> (pool, sfxspace+'pitch_min_to_total')
    amt.afterMaxToBeforeMaxEnergyRatio >> (pool, sfxspace+'pitch_after_max_to_before_max_energy_ratio')
    essentia.run(gen)

    pc = standard.Centroid(range=len(pitch)-1)(pitch)
    pool.set(sfxspace+'pitch_centroid', pc)

def compute(pool, namespace=''):
    # 5th pass: High-level descriptors that depend on others, but we
    #              don't need to stream the audio anymore

    # Average Level
    from level import levelAverage
    levelAverage(pool, namespace)

    # SFX Descriptors
    sfxPitch(pool, namespace)

    # Tuning System Features
    tuningSystemFeatures(pool, namespace)

    # Pool Cleaning (removing temporary descriptors):
    tonalPoolCleaning(pool, namespace)

    # Add missing descriptors which are not computed yet, but will be for the
    # final release or during the 1.x cycle. However, the schema need to be
    # complete before that, so just put default values for these.
    postProcess(pool, namespace)



