import java.nio.file.Path
import java.nio.file.Paths

params.refChannelIndex = 0

// unmix-specific parameters
params.nMosaic = 49
params.unmixMidChunkSize = 512
params.unmixDownsample = 4 // 8
params.overwriteUnmixingMosaics = false

// registration/stitching parameters
params.ashlarMaxShiftUM = 50
params.ashlarFilterSigma = 4  // 4 may be too much, it may register but not too precisely; 2 may be too little and fails to register
params.ashlarTileSize = 1024

String newline = "\n"
String colSep = "\t"

process COLLECT_TILES_BY_SCENE_ROUND {
    tag "scene: ${scene}, round: ${round}"

    input:
        val slide
        tuple \
            val(scene), \
            val(round)

    output:
        tuple \
            val(scene), \
            val(round), \
            path("tiles-${scene}-${round}.txt")

    exec:
        List<Path> tilePaths = slide.getScene(scene).getTilePathsForRound(round)
        Path dst = task.workDir.resolve("tiles-${scene}-${round}.txt")
        file(dst).text = tilePaths.join(newline)
}

process STACK_TILES_BY_SCENE_ROUND {
    if (!workflow.stubRun) {
        conda "./environment.yml"
    }
    tag "scene: ${scene}, round: ${round}"
    memory "16 GB"
    cpus 2

    publishDir(
        path: "${params.logDir}/stack-tiles/",
        enabled: !workflow.stubRun,
        mode: "copy",
        pattern: "stack-tiles-*.dask-performance.html",
    )

    input:
        tuple \
            val(scene), \
            val(round), \
            path(flatfieldPath), \
            path(darkfieldPath), \
            path(tilesPath)

    output:
        tuple \
            val(scene), \
            val(round), \
            path(tilesPath), \
            path("stack-${scene}-${round}.ome.tiff"), emit: stacks
        path("stack-tiles-${scene}-${round}.dask-performance.html")

    script:
        Path darkDir = Paths.get(params.calibrationDir).resolve("dark")
        Path scalesShiftsDir = Paths.get(params.calibrationDir).resolve("cube-alignment").resolve("scales-shifts")
        """
        stack-tiles \
            --tiles-path "${tilesPath}" \
            --flatfield-path "${flatfieldPath}" \
            --darkfield-path "${darkfieldPath}" \
            --dark-dir "${darkDir}" \
            --scales-shifts-dir "${scalesShiftsDir}" \
            --dst "stack-${scene}-${round}.ome.tiff" \
            --dask-report-filename "stack-tiles-${scene}-${round}.dask-performance.html" \
            --n-cpus "${task.cpus}" \
            --memory-limit "${task.memory}"
        """

    stub:
        """
        touch "stack-${scene}-${round}.ome.tiff"
        touch "stack-tiles-${scene}-${round}.dask-performance.html"
        """
}

process COLLECT_STACKS_BY_ROUND {
    tag "round: ${round}"
    input:
        tuple val(round), val(stackPaths)

    output:
        tuple val(round), path("stacks-${round}.txt")

    exec:
        Path dst = task.workDir.resolve("stacks-${round}.txt")
        file(dst).text = stackPaths.join(newline)
}

process MAKE_UNMIXING_MOSAIC_BY_ROUND {
    if (!workflow.stubRun) {
        conda "./environment.yml"
    }
    tag "round: ${round}"

    publishDir(
        path: "${unmixMosaicsDir}",
        mode: "copy",
        enabled: !workflow.stubRun,
        pattern: "unmixing-mosaic-*.tiff"
    )

    input:
        tuple \
            val(round), \
            path(stacksPath)
        val unmixMosaicsDir

    output:
        path "unmixing-mosaic-${round}.tiff"

    script:
        """
        make-unmixing-mosaic \
            --stacks-path "${stacksPath}" \
            --ref-channel "${params.refChannelIndex}" \
            --downsample "${params.unmixDownsample}" \
            --n-tiles "${params.nMosaic}" \
            --mid-chunk-size "${params.unmixMidChunkSize}" \
            --dst "unmixing-mosaic-${round}.tiff"
        """

    stub:
        """
        touch "unmixing-mosaic-${round}.tiff"
        """
}

process SORT_STACKS {
    // needed so that the cycles for each scene are in order
    tag "scene: ${scene}"

    input:
        tuple \
            val(scene), \
            val(rounds), \
            val(tilesPaths), \
            val(stackPaths)

    output:
        tuple \
            val(scene), \
            val(roundsSorted), \
            val(tilesPathsSorted), \
            val(stackPathsSorted)

    exec:
        /*
        This is really ugly; preferably we would like to sort all three lists using the
        first list as a key, but I can't be bothered to make it more complicated than it
        really is
        */
        roundsSorted = rounds.sort({ a, b -> a <=> b})
        tilesPathsSorted = tilesPaths.sort({ a, b -> a.name <=> b.name})
        stackPathsSorted = stackPaths.sort({ a, b -> a.name <=> b.name})
}

process COLLECT_STACKS_BY_SCENE {
    tag "scene: ${scene}"
    input:
        tuple val(scene), val(stackPaths)

    output:
        tuple val(scene), path("stacks-${scene}.txt")

    exec:
        Path dst = task.workDir.resolve("stacks-${scene}.txt")
        file(dst).text = stackPaths.join(newline)
}

process COLLECT_TILES_BY_SCENE {
    tag "scene: ${scene}"

    input:
        tuple \
            val(scene), \
            val(rounds), \
            val(tilesPaths)

    output:
        tuple val(scene), path("tiles-by-round-${scene}.tsv")

    exec:
        List<String> lines = [["round", "tiles-path"].join(colSep)]
        List<List> roundsPaths = [rounds, tilesPaths].transpose()
        roundsPaths = roundsPaths.sort({a, b -> a[0] <=> b[0]})
        for (roundPath in roundsPaths) {
            lines.add(roundPath.join(colSep))
        }
        Path dst = task.workDir.resolve("tiles-by-round-${scene}.tsv")
        file(dst).text = lines.join(newline)
}

process REGISTER_AND_STITCH_SCENE {
    if (!workflow.stubRun) {
        conda "./environment.yml"
    }
    tag "scene: ${sceneName}"

    publishDir(
        path: "${sceneDir}",
        mode: "copy",
        enabled: !workflow.stubRun,
    )

    input:
        val slideName
        tuple \
            val(sceneName), \
            val(sceneDir), \
            path(stacksPath), \
            path(tilesPath)

    output:
        tuple \
            val(sceneName), \
            path("${params.omeTiffFilename}"), \
            path("${params.omeXmlFilename}")

    script:
        /*
        `write-ome-metadata` is here with `register-and-stitch` because the publish operation must happen after both
        commands
        */
        String omeImageName = "{\\\"slide\\\": \\\"${slideName}\\\", \\\"scene\\\": \\\"${sceneName}\\\"}"
        """
        register-and-stitch \
            --output "${params.omeTiffFilename}" \
            --stacks-path "${stacksPath}" \
            --align-channel "${params.refChannelIndex}" \
            --filter-sigma "${params.ashlarFilterSigma}" \
            --maximum-shift "${params.ashlarMaxShiftUM}" \
            --tile-size "${params.ashlarTileSize}"

        write-ome-metadata \
            --stacks-path "${stacksPath}" \
            --tiles-path "${tilesPath}" \
            --ome-tiff-path "${params.omeTiffFilename}" \
            --ome-image-name "${omeImageName}" \
            --ome-xml-path "${params.omeXmlFilename}"
        """

    stub:
        """
        touch "${params.omeTiffFilename}"
        touch "${params.omeXmlFilename}"
        """
}

process MAKE_SCENE_ZARR_FROM_TIFF {
    if (!workflow.stubRun) {
        conda "./environment.yml"
    }
    tag "scene: ${sceneName}"

    publishDir(
        path: "${sceneDir}",
        mode: "copy",
        enabled: !workflow.stubRun,
    )

    input:
        tuple \
            val(sceneName), \
            val(sceneDir), \
            path(omeTiffPath)

    output:
        path "${params.omeZarrDirname}"

    script:
        """
        tiff-to-zarr \
            --src "${omeTiffPath}" \
            --dst "${params.omeZarrDirname}" \
            --tile-size "${params.ashlarTileSize}"
        """

    stub:
        """
        mkdir "${params.omeZarrDirname}"
        """
}

workflow PROCESS_SCENES {
    take:
        slide
        illumProfilesByRound

    main:
        /*
        This way of mapping is unnecessary but lets the programmer know what to expect.
        It won't run if they are switched, but still...
        */
        channel.fromList(slide.getSceneNames()).set { scenesCh }

        COLLECT_TILES_BY_SCENE_ROUND(
            slide,
            scenesCh
                .map { [it, slide.getScene(it).getRoundNames()] }
                .transpose()
                .map { [scene: it[0], round: it[1]] }
                .map { [it.scene, it.round] }
        )

        STACK_TILES_BY_SCENE_ROUND(
            scenesCh
               .combine(illumProfilesByRound)
               .map { [scene: it[0], round: it[1], flatfield: it[2], darkfield: it[3]] }
               .filter { slide.getScene(it.scene).getRoundNames().contains(it.round) }
               .map { [it.scene, it.round, it.flatfield, it.darkfield] }
               .join(COLLECT_TILES_BY_SCENE_ROUND.out, by: [0, 1])
           )

        COLLECT_STACKS_BY_ROUND(
            STACK_TILES_BY_SCENE_ROUND.out.stacks
                .groupTuple(by: 1)  // rounds
                .map { [round: it[1], stacksPath: it[3]] }
                .map { [it.round, it.stacksPath] }
        )
        MAKE_UNMIXING_MOSAIC_BY_ROUND(
            COLLECT_STACKS_BY_ROUND.out,
            slide.unmixMosaicsDir,
        )

        SORT_STACKS(STACK_TILES_BY_SCENE_ROUND.out.stacks.groupTuple(by: 0))
        SORT_STACKS.out
            .map { [
                scene: it[0],
                rounds: it[1],
                tilesPaths: it[2],
                stackPaths: it[3],
            ] }
            .set { roundPropsBySceneCh }

        COLLECT_STACKS_BY_SCENE(
            roundPropsBySceneCh
                .map { [scene: it.scene, stackPaths: it.stackPaths] }
        )
        COLLECT_TILES_BY_SCENE(
            roundPropsBySceneCh
                .map { [scene: it.scene, rounds: it.rounds, tilesPaths: it.tilesPaths ]}
        )

        // prepare all scene metadata in list of tuples
        scenesCh
            .map { [it, slide.getScene(it).path] }
            .join(COLLECT_STACKS_BY_SCENE.out, by: 0)
            .join(COLLECT_TILES_BY_SCENE.out, by: 0)
            .map { [
                scene: it[0],
                sceneDir: it[1],
                stacksPath: it[2],
                tilesPath: it[3]
            ] }
            .set { sceneStacksPathTilesPathCh }
        REGISTER_AND_STITCH_SCENE(
            slide.name,
            sceneStacksPathTilesPathCh
                .map {[
                    it.scene,
                    it.sceneDir,
                    it.stacksPath,
                    it.tilesPath,
                ]}
        )

        MAKE_SCENE_ZARR_FROM_TIFF(
            sceneStacksPathTilesPathCh
                .map { [it.scene, it.sceneDir] }
                .join(REGISTER_AND_STITCH_SCENE.out, by: 0)
                .map { [scene: it[0], sceneDir: it[1], omeTiffPath: it[2]] }
                .map { [it.scene, it.sceneDir, it.omeTiffPath] }
        )
}
