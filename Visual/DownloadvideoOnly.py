import yt_dlp
import os

# SAVE LOCATION
BASE_PATH = "D:\CollectedVideos"

# URLS
SAFE_VIDEOS = {
    "peppa_pig": [
    #    "https://youtu.be/vfMTu__b_xY?si=zeOf7opv1EPs3BL5",
    #     "https://youtu.be/6jKZRaovJEs?si=L_snEjwMA0NZ12Zj",
    #     "https://youtu.be/Mg1eXN4ayrs?si=vyoWfJJ1BHNzZE39",
    #     "https://youtu.be/eDGLkeB9lRk?si=KhayjOa8AJ3sjXzN",
    #     "https://youtu.be/nlLnXOLaSPM?si=Tto4DKEfcNkuLC_-",
    #     "https://youtu.be/bAh_UvgnVl4?si=EyBiGNQku9076lJ-",
    # ],
    # "spongebob": [
    #     "https://youtu.be/yAOU9Yi40EQ?si=fglyY8MbFNnAfVMg",
    #     "",
    #     "https://youtu.be/1P2t-HpfTUI?si=HFUXvDLFp4SGSRLO",
    #     "https://youtu.be/-m6NUoJYxZ0?si=Iy-kER3xxrBA1I-i",
    #     "https://youtu.be/P6RIdVN1kGk?si=Qse-bUPxKdDhbFjP",
    #     "https://youtu.be/LqwhDV6HxIQ?si=nuHAXI1orbgd7wIi",
    # ],
    # "masha_and_bear": [
    #     "https://youtu.be/1BmcE6OFRyE?si=9gneIVwKjl9ekbVx",
    #     "https://youtu.be/kEvA23H0k9U?si=pRml6_At-9gzqp1n",
    #     "https://youtu.be/X6Hur8jbfNo?si=u8SUP0Z6qmUiRICE",
    #     "https://youtu.be/YkE64C4LvMM?si=2BvJ9XIOHgAM110Z",
    #     "https://youtu.be/FXl8HWbTx7E?si=17Rt6VT5TslD0nIm",
    #     "https://youtu.be/YkkacCOkV5s?si=h4cHPwGC8CuCjh4R",
    # ],
    # "pj_masks": [
    #     "https://youtu.be/pYegOSTHe5M?si=HpQHIU4MylAy1X06",
    #     "https://youtu.be/hJyYqjyqCEo?si=_VJ8pRimw6diaTGy",
    #     "https://youtu.be/AAcCWNnP7sQ?si=-zFiizFkUwAanHpt",
    #     "https://youtu.be/zzIMggwbWr4?si=_Zy6IsLti6v_0R2L",
    #     "https://youtu.be/ASKFuc1UXqY?si=oCcRQEYsYEGNeh54",
    #     "https://youtu.be/uur4VweSj5k?si=NdeZ-EfNtgkrD1mf",
    # ],
    # "randomSafe": [
    #     "https://youtu.be/_peoAkmWMb8?si=Qglurvoed0otUDV9",
    #     "https://youtu.be/mvAV3gU0XRE?si=d7zJiv460RJdg32v",
    #     "https://youtu.be/IYT2mqd2qY8?si=itqyZSrZgoUQSUs0",
    #     "https://youtu.be/rnis8lg1DbA?si=AK9JIEwn5BP910kC",
    #     "https://youtu.be/yarRQydQ9eo?si=sOmTr6Dn_oaxQMUU",
    ],
}

UNSAFE_VIDEOS = {
    "south_park": [
        "https://youtu.be/62mv0uFR0Y4?si=coL1a3z4Gf3ZpJYf",
    #     "https://youtu.be/bWf-W5c5j9o?si=3-wz-4uf-0IAEUVS",
    #     "https://youtu.be/LIteRy3hhrE?si=9b0v7miqzcSHRyTI",
    #     "https://youtu.be/lSMTVZ58fvc?si=FJXvejl6MGIW9J0I",
    #     "https://youtu.be/URz-RYEOaig?si=zQvGfJUdet6o3X6Z",
    #     "https://youtu.be/dEcWDA3gAz4?si=w8gkt8jwrVD39OYP",
    #     "https://youtu.be/VWM4ejIfGA4?si=l_BBe11RO4YTPKly",
    #     "https://youtu.be/_1YtDDxbQ8s?si=Bjx3PR5J6XMdtWHs",
    #     "https://youtu.be/m0o3ZIvYEP8?si=pZ2MIf7IaRFQ0MY3",
    #     "https://youtu.be/xy0D5tp1R1Y?si=fQKMCqaUqIPqBtaj",
    #     "https://youtu.be/bSKAWmctzCI?si=uLVXyzuq7OOv7LZQ",
    # ],
    # "family_guy": [
    #     "https://youtu.be/QbB_Nc1_cCY?si=Z5OZygCwncEXWvOg",
    #     "https://youtu.be/2-ksj1AAU6s?si=_1yDjCyJ8NoL1HZr",
    #     "https://youtu.be/dVFm8veHbqc?si=nQgIFF0HX3cUNRW1",
    #     "https://youtu.be/Z7Ra8Wa-WMI?si=L4q_cUve6x_TCyHH",
    #     "https://youtu.be/kE7sjoQtmmQ?si=uAmfnSD-FE-O968X",
    #     "https://youtu.be/rXO58Tuw7hM?si=cavd254VZl6lC7Zq",
    # ],
    # "simpsons": [
    #     "https://youtu.be/0DUoRZJBHoE?si=Bj7Yi83u1kPxNA_c",
    #     "https://youtu.be/Hw3Ko68OQuA?si=xkhJEruORK48EUqu",
    #     "https://youtu.be/3R6IO5lx_Z0?si=i12Osaxbz65UrqGQ",
    #     "https://youtu.be/VbSBiG1Dybg?si=f3Lhs3KNnCvtCRzv",
    #     "https://youtu.be/gFHI5CDlubQ?si=AARvJ6zajD9UHKBk",
    #     "https://youtu.be/-lJEgwzhF7Y?si=orGy1X2gVj7JLhv1",
    # ],
    # "rick_and_monty": [
    #     "https://youtu.be/PPPNUC9_ocQ?si=YugsxfIBNkRD8_WG",
    #     "https://youtu.be/qPL5wLQR9GI?si=wfUzeUy_4rmTcJ-j",
    #     "https://youtu.be/jzV4RdrIfvY?si=9QVySeNsdPYj6b71",
    #     "https://youtu.be/URm_ObHhbrs?si=zWsWV8GJHzlNa3Or",
    #     "https://youtu.be/G3wjw5cIvYU?si=PCOeCRQ21mLxC9ti",
    #     "https://youtu.be/dnkGdGSYNLU?si=AfEdcj3jJmUCZ7Tk",
    # ],
    # "bojack_horseman": [
    #     "https://youtu.be/eatqGb-OOak?si=SmKjH3_22zGiTRI2",
    #     "https://youtu.be/51vxFif765M?si=87BvpYHTxLTFP2es",
    #     "https://youtu.be/DiDN_cdZveI?si=aoyP8Mumz875vY27",
    #     "https://youtu.be/uEGFYhuXD5Q?si=3bYCF0SPejRH_B1g",
    #     "https://youtu.be/qeDGjrbWSOw?si=4Q92M7C8B6G54J-O",
    #     "https://youtu.be/2jawOBAaQJA?si=Hr8XP4ClQLtVv_kz",
    # ],
    # "RandomUnsafe": [
    #     "https://youtu.be/LC-mc-fm3Ak?si=-i9uoKdaeTrsgEwd",
    #     "https://youtu.be/Al60zUQa5PI?si=XBiJJGM4RaSCMDzF",
    #     "https://youtu.be/iWSqrgSfAxM?si=6rbNF7WbITGHek9z",
    #     "https://youtu.be/LC-mc-fm3Ak?si=jcz4kpvYh3BTJgns",
    #     "https://youtu.be/7fh3cR1kRyk?si=gl2az-TJ5BU1iZ3L",
    #     "https://youtu.be/U9k8K9wL_-0?si=mXjDRGnwf927dh-B",
    ],
}


# DOWNLOAD VIDEO
def download_videos(show_name, urls, label):
    output_folder = os.path.join(BASE_PATH, label, show_name)
    os.makedirs(output_folder, exist_ok=True)

    valid_urls = [u.strip() for u in urls if u.strip()]
    if not valid_urls:
        print(f"   No URLs found for {show_name} — skipping")
        return

    print(f"\n  [{show_name}] — {len(valid_urls)} videos → {output_folder}")
    print(f"  {'-'*40}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/bestvideo',
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'cookiefile': r'D:\PythonYouTube\Visual\cookies',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, url in enumerate(valid_urls):
            print(f"\n  Downloading [{i+1}/{len(valid_urls)}]: {url}")
            try:
                ydl.download([url])
                print(f"  Done")
            except Exception as e:
                print(f"  Failed: {e}")

    print(f"\n   {show_name} COMPLETE")


# MAIN
if __name__ == "__main__":
    print("=" * 55)
    print("   VIDEO DOWNLOAD STARTING")
    print(f"   Saving to: {BASE_PATH}")
    print("=" * 55)

    print("\n SAFE VIDEOS")
    print("=" * 55)
    for show_name, urls in SAFE_VIDEOS.items():
        download_videos(show_name, urls, label="safe")

    print("\n  UNSAFE VIDEOS")
    print("=" * 55)
    for show_name, urls in UNSAFE_VIDEOS.items():
        download_videos(show_name, urls, label="unsafe")

    print("\n All downloads complete!")