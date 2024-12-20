import numpy as np

from .common import Layout, Entry, Reserved, FileHandler, get_unit_conversion, RawSurface, read_array
from ..exceptions import CorruptedFileError

LEN_MAGIC = 4

MAGIC_1 = b"\x88\x1b\x03\x6f"
MAGIC_2 = b"\x88\x1b\x03\x70"
MAGIC_3 = b"\x88\x1b\x03\x71"

# (magic, header_format, header_size)
VALID_FORMAT_SPECIFICATIONS = {
    (MAGIC_1, 1, 834),
    (MAGIC_2, 2, 834),
    (MAGIC_3, 3, 4096)
}

INVALID_VALUE_INTENSITY = 65535
INVALID_VALUE_PHASE = 2147483640

# Actually, the resolution depends also on the header format
# However, the only difference is that header formats 2 and 3 also support phase_res of 3.
# So we don't need to actually differentiate the cases.
RESOLUTION_MAP = {
    1: 4096,
    2: 32768,
    3: 131072
}

LAYOUT_HEADER = Layout(
    Entry('header_format', '>H'),
    Entry('header_size', '>I'),
    Entry('swinfo_type', '>h'),
    Entry('swinfo_date', '30s'),
    Entry('swinfo_vers_maj', '>h'),
    Entry('swinfo_vers_min', '>h'),
    Entry('swinfo_vers_bug', '>h'),
    Entry('ac_org_x', '>h'),
    Entry('ac_org_y', '>h'),
    Entry('ac_width', '>H'),
    Entry('ac_height', '>H'),
    Entry('ac_n_buckets', '>H'),
    Entry('ac_range', '>h'),
    Entry('ac_n_bytes', '>I'),
    Entry('cn_org_x', '>h'),
    Entry('cn_org_y', '>h'),
    Entry('cn_width', '>H'),
    Entry('cn_height', '>H'),
    Entry('cn_n_bytes', '>I'),
    Entry('time_stamp', '>i'),
    Entry('comment', '82s'),
    Entry('source', '>h'),
    Entry('intf_scale_factor', '>f'),
    Entry('wavelength_in', '>f'),
    Entry('num_aperture', '>f'),
    Entry('obliquity_factor', '>f'),
    Entry('magnification', '>f'),
    Entry('lateral_res', '>f'),
    Entry('acq_type', '>h'),
    Entry('intens_avg_cnt', '>h'),
    Entry('ramp_cal', '>h'),
    Entry('sfac_limit', '>h'),
    Entry('ramp_gain', '>h'),
    Entry('part_thickness', '>f'),
    Entry('sw_llc', '>h'),
    Entry('target_range', '>f'),
    Entry('rad_crv_veasure_eeq', '<h'),
    Entry('min_mod', '>i'),
    Entry('min_mod_count', '>i'),
    Entry('phase_res', '>h'),
    Entry('min_area', '>i'),
    Entry('discon_action', '>h'),
    Entry('discon_filter', '>f'),
    Entry('connect_order', '>h'),
    Entry('sign', '>h'),
    Entry('camera_width', '>h'),
    Entry('camera_height', '>h'),
    Entry('sys_type', '>h'),
    Entry('sys_board', '>h'),
    Entry('sys_serial', '>h'),
    Entry('inst_id', '>h'),
    Entry('obj_name', '12s'),
    Entry('part_name', '40s'),
    Entry('codev_type', '>h'),
    Entry('phase_avg_cnt', '>h'),
    Entry('sub_sys_err', '>h'),
    Reserved(16),
    Entry('part_ser_num', '40s'),
    Entry('refractive_index', '>f'),
    Entry('rem_tilt_bias', '>h'),
    Entry('rem_fringes', '>h'),
    Entry('max_area', '>i'),
    Entry('setup_type', '>h'),
    Entry('wrapped', '>h'),
    Entry('pre_connect_filter', '>f'),
    Entry('wavelength_in_2', '>f'),
    Entry('wavelength_fold', '>h'),
    Entry('wavelength_in_1', '>f'),
    Entry('wavelength_in_3', '>f'),
    Entry('wavelength_in_4', '>f'),
    Entry('wavelen_select', '8s'),
    Entry('fda_res', '>h'),
    Entry('scan_descr', '20s'),
    Entry('n_fiducials_a', '>h'),
    Entry('fiducials_a', '>14f'),
    Entry('pixel_width', '>f'),
    Entry('pixel_height', '>f'),
    Entry('exit_pupil_diam', '>f'),
    Entry('light_level_pct', '>f'),
    Entry('coords_state', '<i'),
    Entry('coords_x_pos', '<f'),
    Entry('coords_y_pos', '<f'),
    Entry('coords_z_pos', '<f'),
    Entry('coords_x_rot', '<f'),
    Entry('coords_y_rot', '<f'),
    Entry('coords_z_rot', '<f'),
    Entry('coherence_mode', '<h'),
    Entry('surface_filter', '<h'),
    Entry('sys_err_file_name', '28s'),
    Entry('zoom_descr', '8s'),
    Entry('alpha_part', '<f'),
    Entry('beta_part', '<f'),
    Entry('dist_part', '<f'),
    Entry('cam_split_loc_x', '<h'),
    Entry('cam_split_loc_y', '<h'),
    Entry('cam_split_trans_x', '<h'),
    Entry('cam_split_trans_y', '<h'),
    Entry('material_a', '24s'),
    Entry('material_b', '24s'),
    Entry('cam_split_unused', '<h'),
    Reserved(2),
    Entry('dmi_ctr_x', '<f'),
    Entry('dmi_ctr_y', '<f'),
    Entry('sph_dist_corr', '<h'),
    Reserved(2),
    Entry('sph_dist_part_na', '<f'),
    Entry('sph_dist_part_radius', '<f'),
    Entry('sph_dist_cal_na', '<f'),
    Entry('sph_dist_cal_radius', '<f'),
    Entry('surface_type', '<h'),
    Entry('ac_surface_type', '<h'),
    Entry('z_position', '<f'),
    Entry('power_multiplier', '<f'),
    Entry('focus_multiplier', '<f'),
    Entry('rad_crv_vocus_sal_lactor', '<f'),
    Entry('rad_crv_vower_ral_lactor', '<f'),
    Entry('ftp_left_pos', '<f'),
    Entry('ftp_right_pos', '<f'),
    Entry('ftp_pitch_pos', '<f'),
    Entry('ftp_roll_pos', '<f'),
    Entry('min_mod_pct', '<f'),
    Entry('max_inten', '<i'),
    Entry('ring_of_fire', '<h'),
    Reserved(1),
    Entry('rc_orientation', 'B'),
    Entry('rc_distance', '<f'),
    Entry('rc_angle', '<f'),
    Entry('rc_diameter', '<f'),
    Entry('rem_fringes_mode', '>h'),
    Reserved(1),
    Entry('ftpsi_phase_res', 'B'),
    Entry('frames_acquired', '<h'),
    Entry('cavity_type', '<h'),
    Entry('cam_frame_rate', '<f'),
    Entry('tune_range', '<f'),
    Entry('cal_pix_loc_x', '<h'),
    Entry('cal_pix_loc_y', '<h'),
    Entry('n_tst_cal_pts', '<h'),
    Entry('n_ref_cal_pts', '<h'),
    Entry('tst_cal_pts', '<4f'),
    Entry('ref_cal_pts', '<4f'),
    Entry('tst_cal_pix_opd', '<f'),
    Entry('ref_cal_pix_opd', '<f'),
    Entry('sys_serial2', '<i'),
    Entry('flash_phase_dc_mask', '<f'),
    Entry('flash_phase_alias_mask', '<f'),
    Entry('flash_phase_filter', '<f'),
    Entry('scan_direction', 'B'),
    Reserved(1),
    Entry('pre_fda_filter', '>h'),
    Reserved(4),
    Entry('ftpsi_res_factor', '<i'),
    Reserved(8)
)

LAYOUT_HEADER_3 = Layout(
    Reserved(4),
    Entry('films_mode', '>h'),
    Entry('films_reflectivity_ratio', '>h'),
    Entry('films_obliquity_correction', '>f'),
    Entry('films_refraction_index', '>f'),
    Entry('films_min_mod', '>f'),
    Entry('films_min_thickness', '>f'),
    Entry('films_max_thickness', '>f'),
    Entry('films_min_refl_ratio', '>f'),
    Entry('films_max_refl_ratio', '>f'),
    Entry('films_sys_char_file_name', '28s'),
    Entry('films_dfmt', '>h'),
    Entry('films_merit_mode', '>h'),
    Entry('films_h2g', '>h'),
    Entry('anti_vibration_cal_file_name', '28s'),
    Reserved(2),
    Entry('films_fringe_remove_perc', '>f'),
    Entry('asphere_job_file_name', '28s'),
    Entry('asphere_test_plan_name', '28s'),
    Reserved(4),
    Entry('asphere_nzones', '>f'),
    Entry('asphere_rv', '>f'),
    Entry('asphere_voffset', '>f'),
    Entry('asphere_att4', '>f'),
    Entry('asphere_r0', '>f'),
    Entry('asphere_att6', '>f'),
    Entry('asphere_r0_optimization', '>f'),
    Entry('asphere_att8', '>f'),
    Entry('asphere_aperture_pct', '>f'),
    Entry('asphere_optimized_r0', '>f'),
    Entry('iff_state', '<h'),
    Entry('iff_idr_filename', '42s'),
    Entry('iff_ise_filename', '42s'),
    Reserved(2),
    Entry('asphere_eqn_r0', '>f'),
    Entry('asphere_eqn_k', '>f'),
    Entry('asphere_eqn_coef', '>21f'),
    Entry('awm_enable', '<i'),
    Entry('awm_vacuum_wavelength_nm', '<f'),
    Entry('awm_air_wavelength_nm', '<f'),
    Entry('awm_air_temperature_degc', '<f'),
    Entry('awm_air_pressure_mmhg', '<f'),
    Entry('awm_air_rel_humidity_pct', '<f'),
    Entry('awm_air_quality', '<f'),
    Entry('awm_input_power_mw', '<f'),
    Entry('asphere_optimizations', '>i'),
    Entry('asphere_optimization_mode', '>i'),
    Entry('asphere_optimized_k', '>f'),
    Reserved(2),
    Entry('n_fiducials_b', '>h'),
    Entry('fiducials_b', '>14f'),
    Reserved(2),
    Entry('n_fiducials_c', '>h'),
    Entry('fiducials_c', '>14f'),
    Reserved(2),
    Entry('n_fiducials_d', '>h'),
    Entry('fiducials_d', '>14f'),
    Entry('gpi_enc_zoom_mag', '<f'),
    Entry('asphere_max_distortion', '>f'),
    Entry('asphere_distortion_uncert', '>f'),
    Entry('field_stop_name', '12s'),
    Entry('apert_stop_name', '12s'),
    Entry('illum_filt_name', '12s'),
    Reserved(2606)
)


@FileHandler.register_reader(suffix='.dat', magic=(MAGIC_1, MAGIC_2, MAGIC_3))
def read_dat(filehandle, read_image_layers=False, encoding='utf-8'):
    magic = filehandle.read(LEN_MAGIC)
    if magic not in (MAGIC_1, MAGIC_2, MAGIC_3):
        raise CorruptedFileError('Unrecognized magic detected.')
    header = LAYOUT_HEADER.read(filehandle)

    if (magic, header['header_format'], header['header_size']) not in VALID_FORMAT_SPECIFICATIONS:
        raise CorruptedFileError('File is not a valid metropro file!')

    if header['header_format'] == 3:
        header_3 = LAYOUT_HEADER_3.read(filehandle)

    n_points = header['ac_width'] * header['ac_height'] * header['ac_n_buckets']
    if n_points * 2 != header['ac_n_bytes']:
        raise CorruptedFileError('Intensity buffer does not match expected size!')
    image_layers = {}
    if header['ac_n_bytes'] > 0 and read_image_layers:
        intensity_data = read_array(filehandle, dtype='>u2', count=n_points)
        if header['ac_n_buckets'] > 1:
            intensity_data = intensity_data.reshape(header['ac_height'], header['ac_width'], header['ac_n_buckets'])
            for i in range(header['ac_n_buckets']):
                image_layers[f'Intensity_{i}'] = intensity_data[:, :, i].copy()
        else:
            intensity_data = intensity_data.reshape(header['ac_height'], header['ac_width'])
            image_layers['Intensity'] = intensity_data
        if np.nanmax(intensity_data) > header['ac_range']:
            raise CorruptedFileError('Points above specified range detected in intensity data!')
    else:
        filehandle.seek(header['ac_n_bytes'], 1)

    n_points = header['cn_width'] * header['cn_height']
    phase_data = read_array(filehandle, dtype='>i4', count=n_points).reshape(header['cn_height'],
                                                                             header['cn_width']).astype('float64')
    phase_data[phase_data >= INVALID_VALUE_PHASE] = np.nan
    # Scale the phase data from zygo units to meters
    height_data = phase_data * header['intf_scale_factor'] * header['wavelength_in'] * header['obliquity_factor'] / \
                  RESOLUTION_MAP[header['phase_res']]
    height_data = height_data * get_unit_conversion('m', 'um')

    step_x = step_y = header['lateral_res'] * get_unit_conversion('m', 'um')

    metadata = header
    if header['header_format'] == 3:
        metadata.update(header_3)

    return RawSurface(height_data, step_x, step_y, image_layers=image_layers, metadata=metadata)