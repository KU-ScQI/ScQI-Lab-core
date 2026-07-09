import yaml
import keysight.qcs as qcs

def load_address(filename="address.yaml"):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    address = {
        q: {
            line: qcs.Address(
                chassis=addr["chassis"],
                slot=addr["slot"],
                channel=addr["channel"],
            )
            for line, addr in lines.items()
        }
        for q, lines in data.items()
    }
    return address

def get_address_list(address, qubits, line):
    return [address[f"Q{k}"][line]
             for k in qubits.labels]

def channel_mapping(mapper,qubits,address):

    xy_address = get_address_list(address, qubits, "XY")
    readout_awg_address = get_address_list(address, qubits, "RO")
    dnc_address = get_address_list(address, qubits, "DNC")
    dig_address = get_address_list(address, qubits, "DIG")

    xy_awg = qcs.Channels(qubits.labels, "xy_pulse", absolute_phase = False)
    dig = qcs.Channels(qubits.labels, "acquire_channels", absolute_phase= True)
    readout_awg = qcs.Channels(qubits.labels, "readout_channels", absolute_phase = True)

    mapper.add_channel_mapping(xy_awg, xy_address, qcs.InstrumentEnum.M5300AWG)
    mapper.add_channel_mapping(readout_awg, readout_awg_address, qcs.InstrumentEnum.M5300AWG)
    mapper.add_channel_mapping(dig, dig_address, qcs.InstrumentEnum.M5200Digitizer)
    mapper.add_downconverters(dig_address, dnc_address)

def set_lo_frequencies_RO(mapper,qubits,address,freq):
    readout_awg_address = get_address_list(address, qubits, "RO")
    dnc_address = get_address_list(address, qubits, "DNC")

    mapper.set_lo_frequencies(readout_awg_address, freq)
    mapper.set_lo_frequencies(dnc_address, freq)


def set_lo_frequencies_XY(mapper,qubits,address,freq):
    
    xy_address = get_address_list(address, qubits, "XY")

    mapper.set_lo_frequencies(xy_address, freq)


