# Display logical volumes

## lvs -o lv_uuid,lv_name,vg_name,lv_size,seg_size,seg_le_ranges --units k --reportformat json_std

lv_uuid - Unique identifier.
lv_name - Name. LVs created for internal use are enclosed in brackets.
vg_name - Volume group name
lv_size - Size of LV in current units. [size]
seg_size - Size of segment in current units.
seg_le_ranges - Ranges of Logical Extents of underlying devices in command line format.

---

# Display logical volumes filesystems

## lsblk /dev/volume-group/lg\* -o NAME,FSTYPE --json

NAME - device name
FSTYPE - filesystem type

---

# Display Volume groups

## vgs -o vg_uuid,vg_name,vg_size,vg_free,lv_count,pv_count,vg_permissions,pv_name --units k --reportformat json_std

vg_uuid - Unique identifier.
vg_name - Name.
vg_size - Total size of VG in current units.
vg_free - Total amount of free space in current units.
vg_permissions - VG permissions.
pv_name - PV Name

---

# Display Physical volumes

## pvs -o pv_uuid,pv_name,pv_size,pv_free,pv_used,pv_allocatable,dev_size,pv_missing --units k --reportformat json_std

pv_uuid - Unique identifier.
pv_name - Name.
pv_size - Size of PV in current units.
pv_free - Total amount of unallocated space in current units.
pv_used - Total amount of allocated space in current units.
pv_allocatable - Set if this device can be used for allocation.
dev_size - Size of underlying device in current units.
pv_missing - Set if this device is missing in system.

---
