# Columns to select for GBM GRB table
list_grb_table_col = [
                      # Time
                      't90', # 't50',
                      # Flux
                      'flux_1024', 'flux_64', 'flux_batse_1024', 'flux_batse_64', 'flux_batse_256', 'flux_256',
                      # Ampl
                      'pflx_band_ampl', 'pflx_plaw_ampl', 'pflx_sbpl_ampl', 'pflx_comp_ampl',
                      'flnc_band_ampl', 'flnc_plaw_ampl', 'flnc_sbpl_ampl', 'flnc_comp_ampl',
                      # Epeak
                      'pflx_band_epeak', 'pflx_comp_epeak',
                      'flnc_band_epeak',  'flnc_comp_epeak',
                      # Alpha
                      'pflx_band_alpha',
                      'flnc_band_alpha',
                      # Beta
                      'pflx_band_beta', 'flnc_band_beta',
                      # Phtflux
                      'pflx_band_phtflux','pflx_plaw_phtflux', 'pflx_sbpl_phtflux', 'pflx_comp_phtflux',
                      'flnc_band_phtflux', 'flnc_plaw_phtflux', 'flnc_sbpl_phtflux', 'flnc_comp_phtflux',
                      # Phtfluxb
                      'pflx_band_phtfluxb', 'pflx_plaw_phtfluxb','pflx_sbpl_phtfluxb','pflx_comp_phtfluxb',
                      'flnc_band_phtfluxb', 'flnc_plaw_phtfluxb', 'flnc_sbpl_phtfluxb', 'flnc_comp_phtfluxb',
                      # Ergflux
                      'pflx_band_ergflux', 'pflx_plaw_ergflux', 'pflx_sbpl_ergflux', 'pflx_comp_ergflux',
                      'flnc_band_ergflux', 'flnc_plaw_ergflux', 'flnc_sbpl_ergflux', 'flnc_comp_ergflux',
                      # Index
                      'pflx_plaw_index', 'pflx_sbpl_indx1', 'pflx_sbpl_indx2', 'pflx_comp_index',
                      'flnc_plaw_index', 'flnc_sbpl_indx1',  'flnc_sbpl_indx2', 'flnc_comp_index',
                      # Pivot
                      'pflx_plaw_pivot', 'pflx_sbpl_pivot', 'pflx_comp_pivot',
                      'flnc_plaw_pivot', 'flnc_sbpl_pivot', 'flnc_comp_pivot',
                      # Brk
                      'pflx_sbpl_brken', 'pflx_sbpl_brksc',
                      'flnc_sbpl_brken', 'flnc_sbpl_brksc'
                       ]
