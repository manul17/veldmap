// Extracted from master:veldmodules/data-provider/src/manifest.rs

#[cfg(test)]
mod tests {
    use super::*;

    /// Настоящий манифест Sentinel-3 (`S3A_SY_2_AOD…SEN3`), укороченный до
    /// разбираемого: одна единица измерения внутри пакета.
    const SEN3: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1" version="esa-1.0">
  <informationPackageMap>
    <xfdu:contentUnit ID="packageUnit" unitType="Information Package" textInfo="SYNERGY AOD Package">
      <xfdu:contentUnit ID="NTC_AOD_Unit" unitType="Measurement Data Unit" textInfo="AOD">
        <dataObjectPointer dataObjectID="NTC_AOD_Data"/>
      </xfdu:contentUnit>
      <xfdu:contentUnit ID="geoUnit" unitType="Annotation Data Unit" textInfo="Geodetic">
        <dataObjectPointer dataObjectID="geodeticData"/>
      </xfdu:contentUnit>
    </xfdu:contentUnit>
  </informationPackageMap>
  <dataObjectSection>
    <dataObject ID="geodeticData">
      <byteStream mimeType="application/x-netcdf" size="900">
        <fileLocation locatorType="URL" href="geodetic_an.nc"/>
      </byteStream>
    </dataObject>
    <dataObject ID="NTC_AOD_Data">
      <byteStream mimeType="application/x-netcdf" size="35064511">
        <fileLocation locatorType="URL" href="NTC_AOD.nc"/>
        <checksum checksumName="MD5">6cfc77db3ed681308a24a95e703a48d6</checksum>
      </byteStream>
    </dataObject>
  </dataObjectSection>
</xfdu:XFDU>"#;

    /// Манифест Sentinel-1: тот же XFDU, но путь ведёт в подкаталог и записан
    /// от текущего места.
    const SAFE: &str = r#"<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1">
  <informationPackageMap>
    <xfdu:contentUnit unitType="SAFE Archive Information Package">
      <xfdu:contentUnit unitType="Measurement Data Unit" repID="s1Level2MeasurementSchema">
        <dataObjectPointer dataObjectID="measurementData1"/>
      </xfdu:contentUnit>
      <xfdu:contentUnit unitType="Metadata Unit" repID="s1Level2ProductSchema">
        <dataObjectPointer dataObjectID="productAnnotation"/>
      </xfdu:contentUnit>
    </xfdu:contentUnit>
  </informationPackageMap>
  <dataObjectSection>
    <dataObject ID="measurementData1" repID="s1Level2MeasurementSchema">
      <byteStream mimeType="application/octet-stream" size="41769268">
        <fileLocation locatorType="URL" textInfo="Measurement" href="./measurement/s1c-ew-ocn-hh-001.nc"/>
      </byteStream>
    </dataObject>
    <dataObject ID="productAnnotation">
      <byteStream mimeType="text/xml" size="1024">
        <fileLocation locatorType="URL" href="./annotation/s1c-ew-ocn-hh-001.xml"/>
      </byteStream>
    </dataObject>
  </dataObjectSection>
</xfdu:XFDU>"#;

    /// Названо ровно измерение — и ни геометрия, ни аннотация, хотя лежат они
    /// в том же разделе объектов и в алфавите стоя́т раньше.
    #[test]
    fn the_manifest_names_the_measurement_and_nothing_else() {
        assert_eq!(measurements(SEN3.as_bytes()), vec!["NTC_AOD.nc".to_string()]);
        assert_eq!(
            measurements(SAFE.as_bytes()),
            vec!["measurement/s1c-ew-ocn-hh-001.nc".to_string()]
        );
    }

    /// Не разобралось — значит «не знаю», а не «измерений нет»: половина
    /// ответа хуже отсутствия ответа, потому что на неё положились бы.
    #[test]
    fn a_broken_manifest_says_nothing() {
        assert!(measurements(b"").is_empty());
        assert!(measurements(b"<xfdu:XFDU><informationPackageMap>").is_empty());
        assert!(measurements(b"{\"json\": true}").is_empty());
        // Единица есть, а объекта, на который она указывает, нет.
        let dangling = SEN3.replace("NTC_AOD_Data\">", "otherData\">");
        assert!(measurements(dangling.as_bytes()).is_empty());

        // Оборванное тело — тот же ответ, и это главное: разбор до `Eof`
        // доходит без единой ошибки, и половина названных измерений выглядела
        // бы целым ответом. Режем на границе элементов, у самого конца.
        let cut = SEN3.rfind("</xfdu:XFDU>").expect("корень закрыт");
        assert!(
            measurements(SEN3[..cut].as_bytes()).is_empty(),
            "оборванный манифест не называет ничего"
        );
        // А целый — называет.
        assert!(!measurements(SEN3.as_bytes()).is_empty());
    }

    /// Манифест ищется в корне продукта, а не где придётся: у Sentinel-2 такой
    /// же файл лежит в каждой грануле и описывает её одну.
    #[test]
    fn only_the_manifest_at_the_root_counts() {
        let keys: Vec<String> = [
            "eodata/…/S2C_MSIL2A.SAFE/GRANULE/L2A_T40WFC/MTD_TL.xml",
            "eodata/…/S2C_MSIL2A.SAFE/GRANULE/L2A_T40WFC/manifest.safe",
            "eodata/…/S2C_MSIL2A.SAFE/manifest.safe",
        ]
        .iter()
        .map(|path| path.to_string())
        .collect();
        assert_eq!(
            key("eodata/…/S2C_MSIL2A.SAFE", &keys),
            Some(&"eodata/…/S2C_MSIL2A.SAFE/manifest.safe".to_string())
        );
        assert_eq!(key("eodata/…/S2C_MSIL2A.SAFE/", &keys), key("eodata/…/S2C_MSIL2A.SAFE", &keys));
        assert_eq!(key("eodata/…/X.SEN3", &keys), None);
    }
}
