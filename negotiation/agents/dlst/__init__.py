"""DLST tactical sub-strategy components (strategy template, bidding, acceptance, opponent model).

These modules implement the DLST-ANESIA strategy machinery as-is: a
continuous-action DDPG actor outputs strategy template parameters, which are
then translated into concrete bundle offers/accept decisions by the strategy
classes in this package.
"""
