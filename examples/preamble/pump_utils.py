import time
from windfreak import SynthHD


class PumpController:
    _instances = {}

    def __new__(cls, port, name=None):
        if port in cls._instances:
            return cls._instances[port]

        obj = super().__new__(cls)
        cls._instances[port] = obj
        return obj

    def __init__(self, port, name=None):
        if getattr(self, "_initialized", False):
            return

        self.port = port
        self.name = name or str(port)
        self.synth = None

        # 채널별 마지막 freq/power 저장
        self._last_settings = {
            0: {"freq": None, "power": None},
            1: {"freq": None, "power": None},
        }

        self._initialized = True

    def connect(self, max_wait_minutes=10, retry_interval=60):
        if self.synth is not None:
            print(f"{self.name}: already connected")
            return self

        start = time.time()
        attempt = 0

        while True:
            attempt += 1
            try:
                synth = SynthHD(self.port)
                synth.init()
                self.synth = synth

                if attempt > 1:
                    print(f"{self.name}: Windfreak 연결 성공! ({attempt}번째 시도)")

                return self

            except Exception as e:
                msg = str(e)
                is_busy = "BUSY" in msg or "in use" in msg

                print(f"[debug] {self.name} attempt {attempt}: {type(e).__name__}: {e}")

                if not is_busy:
                    raise RuntimeError(
                        f"{self.name}: Windfreak 연결 실패. "
                        f"브릿지/IP/장비 상태 확인 필요: {e}"
                    )

                elapsed_min = (time.time() - start) / 60
                if elapsed_min >= max_wait_minutes:
                    raise RuntimeError(
                        f"{self.name}: Windfreak가 {max_wait_minutes}분 동안 계속 사용 중입니다. "
                        f"(마지막 에러: {e})"
                    )

                print(
                    f"⏳ {self.name}: Windfreak 사용 중입니다. "
                    f"{retry_interval}초 후 재시도합니다. "
                    f"(경과: {elapsed_min:.1f}분 / 최대 {max_wait_minutes}분)"
                )
                time.sleep(retry_interval)

    def _check_connected(self):
        if self.synth is None:
            raise RuntimeError(
                f"{self.name}: SynthHD가 아직 초기화되지 않았습니다. "
                f"먼저 .connect()를 실행하세요."
            )

    def _check_channel(self, ch):
        if ch not in [0, 1]:
            raise ValueError("channel must be 0 or 1")

    def set(self, ch, freq=None, power=None):
        self._check_connected()
        self._check_channel(ch)

        # 입력된 값이 있으면 마지막 값 갱신
        if freq is not None:
            self._last_settings[ch]["freq"] = float(freq)

        if power is not None:
            self._last_settings[ch]["power"] = float(power)

        # 입력이 생략된 경우 기존 값 사용
        freq_to_set = self._last_settings[ch]["freq"]
        power_to_set = self._last_settings[ch]["power"]

        if freq_to_set is None:
            raise ValueError(
                f"{self.name}: channel {ch}의 freq가 아직 정의되지 않았습니다. "
                f"처음에는 freq를 입력해야 합니다."
            )

        if power_to_set is None:
            raise ValueError(
                f"{self.name}: channel {ch}의 power가 아직 정의되지 않았습니다. "
                f"처음에는 power를 입력해야 합니다."
            )

        self.synth[ch].power = power_to_set
        self.synth[ch].frequency = freq_to_set

        return self

    def on(self, ch):
        self._check_connected()
        self._check_channel(ch)

        self.synth[ch].enable = True
        time.sleep(0.5)

        return self

    def off(self, ch):
        self._check_connected()
        self._check_channel(ch)

        self.synth[ch].enable = False
        return self

    def set_on(self, ch, freq=None, power=None):
        self.set(ch, freq=freq, power=power)
        self.on(ch)
        return self

    def on_all(self):
        self._check_connected()

        self.synth[0].enable = True
        self.synth[1].enable = True
        time.sleep(0.5)

        return self

    def off_all(self):
        self._check_connected()

        self.synth[0].enable = False
        self.synth[1].enable = False
        return self

    def close(self):
        if self.synth is not None:
            self.off_all()
            self.synth.close()
            self.synth = None

        return self

    @classmethod
    def close_all(cls):
        for pump in cls._instances.values():
            pump.close()

    @classmethod
    def get(cls, port):
        return cls._instances.get(port)


# ============================================================
# Example usage
# ============================================================

# # 1. Single Windfreak device
# port = "COM3"
# pump = PumpController(port=port, name="pump").connect()
#
# # Set channel 0
# pump.set(ch=0, freq=6.5e9, power=-10)
# pump.on(ch=0)
#
# # Set channel 1
# pump.set(ch=1, freq=7.1e9, power=-15)
# pump.on(ch=1)
#
# # Turn off one channel
# pump.off(ch=0)
#
# # Turn off both channels
# pump.off_all()
#
# # Close connection
# pump.close()


# # 2. Set and turn on in one line
# pump.set_on(ch=0, freq=6.5e9, power=-10)
# pump.set_on(ch=1, freq=7.1e9, power=-15)


# # 3. Multiple Windfreak devices
# pumps = {
#     "pump1": PumpController(port="COM3", name="pump1").connect(),
#     "pump2": PumpController(port="COM4", name="pump2").connect(),
# }
#
# pumps["pump1"].set_on(ch=0, freq=6.5e9, power=-10)
# pumps["pump1"].set_on(ch=1, freq=7.1e9, power=-15)
#
# pumps["pump2"].set_on(ch=0, freq=5.8e9, power=-12)
# pumps["pump2"].set_on(ch=1, freq=6.2e9, power=-13)
#
# # Turn off and close all devices
# for p in pumps.values():
#     p.off_all()
#     p.close()


# # 4. Safe behavior against duplicate connection
# pump1 = PumpController(port="COM3", name="pump").connect()
# pump2 = PumpController(port="COM3", name="pump").connect()
#
# # Since the port is the same, pump1 and pump2 are the same object.
# print(pump1 is pump2)  # True
#
# pump1.close()


# # 5. Retrieve existing controller by port
# pump = PumpController.get("COM3")
#
# if pump is not None:
#     pump.off_all()
#     pump.close()


# # 6. Close all existing PumpController instances
# PumpController.close_all()