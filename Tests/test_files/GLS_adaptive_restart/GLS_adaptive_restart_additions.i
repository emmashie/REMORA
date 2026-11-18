# Move the patch after the first 10-second step. A restart from step 2 must
# reproduce all checkpoint fields at step 3, including the GLS history.
remora.n_cell = 20 24 12
remora.max_step = 3
remora.fixed_dt = 10
remora.gls_stability_type = canuto_a
remora.plot_int = 3
remora.check_int = 2
amr.regrid_int = 1
remora.refinement_indicators = seam shifted
remora.seam.end_time = 5
remora.shifted.start_time = 5
remora.shifted.max_level = 1
remora.shifted.in_box_lo = 25000. 75000. -50.
remora.shifted.in_box_hi = 75000. 225000. 0.
test.case = regrid
